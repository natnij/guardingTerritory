package territoryGame;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

abstract class Defender {

	String name;
    int[] Dx;
    int[] Dy;
    int[] DRx; // defender range x
    int[] DRy; // defender range y
    double CAPTUREDIST;
    
    protected int B_WIDTH;
    protected int B_HEIGHT;
    protected final double CAPTURETOL = 1E-3;
    protected final double PRECISION = 1E-3;
    protected final double smallNr = -1E6;
    protected final double ZERO = 1E-15;
    protected boolean HITWALL = false;
    int BURNIN = 50; // number of early time steps to keep learning rate stable.
    boolean winner = false;
    
    int Cx; // territory center
    int Cy; // territory center
    int[] Tx; // territory border
    int[] Ty;
    int[] t; // nearest territory border to invader
    double[] inputX; // input vector.
    int[] inputBucket; // number of discretized values for each input, same length as inputX.
    int M; // number of rules.
    double[] inputLower; // lower limit of input space, same length as inputBucket.
    double[] inputUpper; // upper limit of input space, same length as inputBucket.
    double[] MFparamsLower;
    double[] MFparamsUpper;
    Combination<Integer> inputCombination; // combination of discretized values of inputBucket.
    
    double distToInvader;
    double previousDistToInvader; // distance to invader at previous time step.
    double bestDistToInvader; // closest distance to invader during the game.
    double distInvaderToTerritory;
    double distToTerritory;
    double previousDistToTerritory;
    double previousAngle = 0.0;
    
    double[] actions; // action space.
    double[] phi_l; // membership degree for each rule l. length=M.
    double U = 0.0; // defuzzified action value from the continuous action space.    
    String MF = "triangular";
    List<Double[][]> MF_params; // length of list: number of params. row: nr of rules. col: nr. of inputs. for each param a table. currently only two params for each type of MF (e.g. a and b for triangular, mean and stddev for gaussian).

    double epsilonInit = 0.5; // initial epsilon-greedy for choosing actions.
    double[] epsilon; // epsilon value for each rule based on number of times the rule is visited.
    double gamma = 0.9; // time discount factor for Q values.
    int[] updateRulesCount; // number of times the rule is updated. to avoid fixing actions for rules which are never visited.
    int[] bestUpdateRulesCount; // count in accordance with the recorded best actions (closest to invader).
    
	Defender(String defenderName, int ALL_DOTS, int b_width, int b_height, double capturedist, 
			int Dx0, int Dy0, int cx, int cy, int[] tx, int[] ty, int[] Ix0, int[] Iy0, int[] rulescount,
			String mf){
	    name = defenderName;
	    MF = mf;
		Dx = new int[ALL_DOTS];
	    Dy = new int[ALL_DOTS];
        Dx[0] = Dx0;
        Dy[0] = Dy0;
        DRx = new int[360]; // defender range x
        DRy = new int[360]; // defender range y
        CAPTUREDIST = capturedist;
        locateRange();
        
	    B_WIDTH = b_width;
	    B_HEIGHT = b_height;
	    Cx = cx;
	    Cy = cy;
	    Tx = tx.clone();
	    Ty = ty.clone();
	    
	    initGame(Ix0,Iy0, rulescount);
	    initMF();
	}
	
	/**
	 * 
	 * @param Ix0: array of invader x positions in current time step (in case of multiple invaders).
	 * @param Iy0: array of invader y positions in current time step.
	 */
	protected void initGame(int[] Ix0, int[] Iy0, int[] rulescount) {
		
		inputX = new double[]{0.0,0.0};
	    inputBucket = new int[]{8,8}; // by default two inputs, each with 8 discrete values ranged from -pi to pi.
	    inputLower = new double[]{-Math.PI,-Math.PI};
	    inputUpper = new double[]{Math.PI,Math.PI};
    	getCombination();
    	
    	M = inputCombination.size();
	    phi_l = new double[M];
	    if (Arrays.stream(rulescount).sum()==0){
		    updateRulesCount = new int[M];
		    bestUpdateRulesCount = new int[M];
	    } else {
	    	updateRulesCount = rulescount.clone();
	    	bestUpdateRulesCount = rulescount.clone();
	    }

	    epsilon = new double[M];
	    Arrays.fill(epsilon, epsilonInit);
	    	    
	    observe(Ix0, Iy0);
	}
    
    /**
     * basis for calculating membership degree \Phi for each rule l.
     * each rule concerns a combination of input variables.
     */
	protected void getCombination() {
		List<List<Integer>> allArrays = new ArrayList<List<Integer>>();
		for (int i=0;i<inputBucket.length;i++){
			List<Integer> oneArray = IntStream.range(0, inputBucket[i]).boxed().collect(Collectors.toList());
			allArrays.add(oneArray);
		}
        inputCombination = new Combination<Integer>(allArrays);
	}
	
	/**
	 * for drawing the circle of effective capture range.
	 */
	protected void locateRange() {
    	for (int theta=0;theta<DRx.length;theta++) {
    		DRx[theta] = Math.round((float) (Dx[0] + CAPTUREDIST * Math.cos(2 * Math.PI / 360 * theta)));
    		DRy[theta] = Math.round((float)  (Dy[0] + CAPTUREDIST * Math.sin(2 * Math.PI / 360 * theta)));
    	}
	}
	
	public void move(int dots, int Ox, int Oy) {
        for (int z = dots; z > 0; z--) {
            Dx[z] = Dx[(z - 1)];
            Dy[z] = Dy[(z - 1)];
        }
        
        calDefenderNEPos(Ox, Oy);
        
        if (Dx[0] >= B_WIDTH) Dx[0] = B_WIDTH - 1;
        if (Dx[0] < 0) Dx[0] = 0;
        if (Dy[0] >= B_HEIGHT) Dy[0] = B_HEIGHT - 1;
        if (Dy[0] < 0) Dy[0] = 0;
        
        locateRange();
	}
	
	public void move(int dots, int[] Ix0, int[] Iy0) {
		
        for (int z = dots; z > 0; z--) {
            Dx[z] = Dx[(z - 1)];
            Dy[z] = Dy[(z - 1)];
        }
        
        calDefenderPos(dots, Ix0, Iy0);
        locateRange();
    }
	
	protected void calDefenderPos(int dots, int[] Ix0, int[] Iy0) {
		
	}
	
    protected void runOneStep() {
    	
//    	if (doubleEqual(Math.abs(inputX[0]-inputX[1]),Math.PI,CAPTURETOL)){
//		// stay put
//    	} else {
	       	if (doubleEqual(Math.abs(U),Math.PI/2,PRECISION)) {
	    		Dx[0] = Dx[0];
	    		Dy[0] = (int)(Math.signum(U) + Dy[0]);
	    	} else {
	    		
	    		int signX = ((U > -Math.PI / 2) && (U < Math.PI / 2)) ? 1 : -1;
	    		int signY = (U < 0) ? 1 : -1;
	    		
	    		double deltaX2 = 1.0 / (Math.pow(Math.tan(U),2) + 1.0);
	    		float deltaX = (float) Math.sqrt(deltaX2);
	    		float deltaY = (float) Math.sqrt(1-deltaX2);
	        	Dx[0] = Dx[0] + signX * Math.round(deltaX);
	        	Dy[0] = Dy[0] + signY * Math.round(deltaY);
	        }
	    	
	        HITWALL = false;
	        if (Dx[0] >= B_WIDTH) {
	        	Dx[0] = B_WIDTH - 1;
	        	HITWALL = true;
	        }
	        if (Dx[0] < 0) {
	        	Dx[0] = 0;
	        	HITWALL = true;
	        }
	        if (Dy[0] >= B_HEIGHT) {
	        	Dy[0] = B_HEIGHT - 1;
	        	HITWALL = true;
	        }
	        if (Dy[0] < 0) {
	        	Dy[0] = 0;
	        	HITWALL = true;
	        }
//    	}
    }
    
	/**
	 * calculate inputs: angle to the invader and to the territory.
	 * @param Ix0: array of invader x positions in current time step (in case of multiple invaders).
	 * @param Iy0: array of invader y positions in current time step.
	 */
    protected void observe(int[] Ix0, int[] Iy0) {
    	
    	// choose the closest invader to defend against
    	// TODO: find a better strategy, also consider what other defenders are doing.
    	ValContainer<Double> dist = new ValContainer<Double>(calculateDist(Dx[0],Dy[0],Ix0[0],Iy0[0]));
    	ValContainer<Integer> idx = new ValContainer<Integer>(0);
    	for (int i=1; i<Ix0.length; i++) {
    		double newDist = calculateDist(Dx[0],Dy[0],Ix0[i],Iy0[i]);
    		if (newDist < dist.getVal()) {
    			dist.setVal(newDist);
    			idx.setVal(i);
    		}
    	}
    	
    	int Ix = Ix0[idx.getVal()];
    	int Iy = Iy0[idx.getVal()];
    	distToInvader = dist.getVal();
    	
    	calNearestBorder(Ix, Iy);
    	distToTerritory = calculateDist(Dx[0],Dy[0],t[0],t[1]);
    	
    	distInvaderToTerritory = calculateDist(Ix,Iy,Cx,Cy);
    	
    	inputX[0] = calculateAngle(Dx[0], Dy[0], Ix, Iy);
    	inputX[1] = calculateAngle(Dx[0], Dy[0], t[0], t[1]);
    }
    
    protected void calNearestBorder(int Ix, int Iy) {
    	t = new int[]{Tx[0],Ty[0]};
    	ValContainer<Double> dist = new ValContainer<Double>(distInvaderToTerritory);
    	for (int i=1;i<Tx.length;i++) {
    		double newDist = calculateDist(Ix, Iy, Tx[i], Ty[i]);
    		if (newDist < dist.getVal()){
    			dist.setVal(newDist);
    			t[0] = Tx[i];
    			t[1] = Ty[i];
    		}
    	}
    }
    
    protected double calculateAngle(int originX, int originY, int destX, int destY) {
    	if (destX == originX && destY <= originY){
    		return Math.PI / 2;
    	} else if (destX == originX && destY > originY) {
    		return -Math.PI / 2;
    	} else {
    		// Jpanel y-axis points downwards
    		double theta = Math.atan((1.0 * originY - 1.0 * destY) / (1.0 * destX - 1.0 * originX));
    		if(Math.signum(destX-originX)<0) {
    			if (Math.signum(theta)>0) {
    				theta = theta - Math.PI;
    			}else {
    				theta = theta + Math.PI;
    			}
    		}
    		if (doubleEqual(theta,Math.PI,PRECISION))
    			theta = -theta; // -pi is the same as pi. however the membership function only includes -pi values.   		
    		return theta;
    	}   	
    }
    
    protected double calculateDist(int originX, int originY, int destX, int destY) {
    	return Math.sqrt(Math.pow(originX - destX, 2) + Math.pow(originY - destY, 2));
    }
    
    /**
     * http://www.dma.fi.upm.es/recursos/aplicaciones/logica_borrosa/web/fuzzy_inferencia/funpert_en.htm
     * @param a: lower limit
     * @param b: upper limit
     * @param x: input value
     * @return value of membership function F(x)
     */
    protected double calculateTriangularMF(double a, double b, double x) {
    	double m = (a+b)/2;
    	if (x <= a) {
    		return 0;
    	} else if (x <= m) {
    		return (x-a)/(m-a);
    	} else if (x <= b) {
    		return (b-x)/(b-m);
    	}else {
    		return 0;
    	}
    }
    /**
     * calculates gaussian membership function given x.
     * @param mean: fuzzy set mean
     * @param sigma: fuzzy set stddev
     * @param x: input
     * @return value of the membership function with given input.
     */
    protected double calculateGaussianMF(double mean, double sigma, double x) {
    	return Math.exp( -Math.pow( (x-mean)/sigma, 2 ) );
    }
    
    /**
     * @param x: input vector. same length as inputBucket.
     * @param l: index of the rule in question
     * @return membership degrees \Phi_l (altogether M=n1*n2...*ni rules, l \in M)
     */
    protected double calculatePhi_l_numerator(int l, double[] x, List<Double[][]> params) {
    	List<Integer> combi = inputCombination.get(l);
    	ValContainer<Double> numerator = new ValContainer<Double>(1.0);
    	    	
    	for (int i=0; i<combi.size(); i++) {
    		ValContainer<Double> mf = new ValContainer<Double>(0.0);	
    		if (MF.equals("triangular")) {
    			Double[][] MF_param1 = (Double[][]) copyArray(Double.class, params.get(0));
    			Double[][] MF_param2 = (Double[][]) copyArray(Double.class, params.get(1));
    			mf.setVal(calculateTriangularMF(MF_param1[l][i], MF_param2[l][i], x[i]));
    		} else if (MF.equals("gaussian")) {
    			Double[][] MF_param1 = (Double[][]) copyArray(Double.class, params.get(0));
    			Double[][] MF_param2 = (Double[][]) copyArray(Double.class, params.get(1));
    			mf.setVal(calculateGaussianMF(MF_param1[l][i], MF_param2[l][i], x[i]));
    		}
    		numerator.setVal(numerator.getVal() * mf.getVal());
    	}
    	if (Math.abs(numerator.getVal()) <= ZERO)
    		numerator.setVal(0.0);
    	
    	return numerator.getVal();
    }

    /**
     * 
     * @param phiInput: the vector to update.
     * @param x: input vector.
     * @param params: the membership function parameters.
     * @return \Phi value for all M rules.
     */
    protected double[] calculatePhi_l(double[] phiInput, double[] x, List<Double[][]> params) {
    	double[] phi = phiInput.clone();
    	double[] numerators = new double[M];
    	ValContainer<Double> denominator = new ValContainer<Double>(0.0);
    	for (int l=0; l<M; l++){
    		numerators[l] = calculatePhi_l_numerator(l, x, params);
    		if (numerators[l] != 0) {
    			// increase the rule's number of times visited. the higher the count, 
    			// the less likely to explore actions associated to the rule.
    			updateRulesCount[l] ++ ; 
    		}
    		denominator.setVal(denominator.getVal() + numerators[l]);
    	}
    	if (denominator.getVal()!=0.0){
    		phi = DoubleStream.of(numerators).map(d->d/denominator.getVal()).toArray();
    	}
    	return phi;
    }
    
    protected double calculateRewardShaping() {
    	// difference between invader angle and territory border angle (with regard to the defender).
    	// best would be pi degrees apart (i.e. on different sides of the defender),
    	// in which case the angle is 0. any other inputs will result in a negative angle.
    	double angle = - Math.abs(Math.PI - Math.abs(inputX[0]-inputX[1])) / Math.PI;
    	double angleReward = angle - previousAngle; // positive only when angle approaches 0.
    	
    	// distance between invader and defender.
    	double distReward;
    	if (Math.signum(angleReward) >= 0) {
    		distReward = (previousDistToInvader - distToInvader) *5; //+ (distToTerritory - previousDistToTerritory);
    	} else {
    		distReward = 1.0;
    	}
    	previousAngle = angle;
    	previousDistToInvader = distToInvader;
    	previousDistToTerritory = distToTerritory;
    	
    	return (angleReward + distReward) / 6;
    }
    
    protected void initMF(List<Double[][]> params) {
    	if (params.get(0)[0][0].equals(smallNr)) {
    		initMF();
    	} else {
    		MF_params = new ArrayList<Double[][]>();
    		for (int i=0;i<params.size();i++) {
    			MF_params.add((Double[][])copyArray(Double.class, params.get(i)));
    		}
    	}
    	if (MF.equals("gaussian")) {
    	    MFparamsLower = new double[]{inputLower[0],0.01}; // mean, stddev limits
    	    MFparamsUpper = new double[]{inputUpper[0],(inputUpper[0] - inputLower[0]) / inputBucket[0]}; // mean, stddev limits
    	} else {
    		MFparamsLower = new double[]{inputLower[0],inputLower[0]};
    		MFparamsUpper = new double[]{inputUpper[0],inputUpper[0]};
    	}
    }
    
    protected void initMF() {
    	MF_params = new ArrayList<Double[][]>();
    	Double[][] param1 = new Double[M][inputX.length];
    	Double[][] param2 = new Double[M][inputX.length];
    	for (int l=0;l<M;l++) {
        	List<Integer> combi = inputCombination.get(l);
        	for (int i=0; i<combi.size(); i++) {
        		int bucket = combi.get(i);
    			double step = (inputUpper[i] - inputLower[i]) / inputBucket[i];
        		if (MF.equals("triangular")) {
            		param1[l][i] = inputLower[i] + (bucket - 1) * step; // lower bound of triangular MF
            		param2[l][i] = inputLower[i] + (bucket + 1) * step; // upper bound of triangular MF
        		} else if (MF.equals("gaussian")) {
        			param1[l][i] = inputLower[i] + bucket * step; // mean of gaussian MF
        			param2[l][i] = step / 2; // stddev of gausian MF
        		}
        	}
    	}
    	MF_params.add(param1);
    	MF_params.add(param2);
    	
    	if (MF.equals("gaussian")) {
    	    MFparamsLower = new double[]{inputLower[0],0.01}; // mean, stddev limits
    	    MFparamsUpper = new double[]{inputUpper[0],(inputUpper[0] - inputLower[0]) / inputBucket[0]}; // mean, stddev limits
    	} else {
    		MFparamsLower = new double[]{inputLower[0],inputLower[0]};
    		MFparamsUpper = new double[]{inputUpper[0],inputUpper[0]};
    	}
    }
    
    
    protected void updateBestParams_basic() {
		bestDistToInvader = distToInvader;
		bestUpdateRulesCount = updateRulesCount.clone();
    }
    
    /**
     * update epsilon value (decrease probability of selecting suboptimal 
     * actions as game proceeds).
     * @param dots
     */
    protected void updateEpsilon(int dots) {
    	if (dots > BURNIN) {
    		epsilon = IntStream.range(0, M).mapToDouble(idx -> epsilon[idx] / Math.log(1.0 * updateRulesCount[idx])).toArray();
    	}
    }
    
    /**
     * utility function to create ascending list of double values with equal interval, 
     * starting from lower limit, excluding upper limit.
     * 
     * @param lower: lower limit (starting point included)
     * @param upper: upper limit (excluded)
     * @param nrSteps: array length.
     * @return array of double.
     */
    protected double[] createDoubleRange(double lower, double upper, int nrSteps) {
    	double stepLen = (upper - lower) / nrSteps;
    	return IntStream.range(0, nrSteps).mapToDouble(x->lower + x * stepLen).toArray();
    }
    
    /**
     * utility function to compare doubles. 
     * @param a
     * @param b
     * @return
     */
    protected boolean doubleEqual(double a, double b, double precision) {
    	double c = a / b;
    	return (Math.abs(c - 1.0) < precision);
    }
    
    protected <T> void printArray(List<T> al, String remark) {
    	System.out.print(remark + " ");
    	for(int i=0;i<al.size();i++)
    		System.out.print(al.get(i) + " ");
    	System.out.print("\n");
    }
    
    protected void calDefenderNEPos(int Ox, int Oy) {
    	double dist = Math.sqrt(Math.pow(Dx[0]-Ox,2)+Math.pow(Dy[0]-Oy,2));
    	Dx[0] = Dx[0] + Math.round((float) (1/dist * (Ox-Dx[0])));
    	Dy[0] = Dy[0] + Math.round((float) (1/(dist) * (Oy-Dy[0])));
    }
    
    
    /**
     * a very ugly generic array deep-copy method using reflection to get around type erasure.
     * https://docs.oracle.com/javase/tutorial/reflect/special/arrayInstance.html
     * @param klazz: class of the !!componentType!! if you passed in the type of the object(e.g. 2-dim array) you'll end up having 4 dimensions...
     * @param original: the original 2d array of generic type.
     * @return a copied 2d array of the same type as original.
     */
    protected <T> T[][] copyArray(Class<? extends T> klazz, T[][] original) {
    	
		@SuppressWarnings("unchecked")
		T[][] newArray = (T[][]) Array.newInstance(klazz, original.length,original[0].length);
    	
    	for (int row=0;row<original.length;row++)
    		newArray[row] = original[row].clone();
    	
    	return newArray;
    }
    
    /**
     * update param value (decrease learning rate).
     * @param param: the parameter to be updated (e.g. different learning rates)
     * @param dots: time step
     * @return new parameter value.
     */
    protected double updateParam(double param, int dots) {
    	double newParam = param;
    	if (dots > BURNIN) {
    		newParam = param / Math.log(1.0 * dots);
    	}
    	if (Math.abs(newParam) <= ZERO)
    		return 0.0;
    	return newParam;
    }
    
    protected void normalizeRows(Double[][] param) {
    	for (int l=0; l<param.length; l++) {
    		Double tmp = Arrays.stream(param[l]).mapToDouble(Double::doubleValue).sum();
    		if (tmp==0) {
    			Arrays.fill(param[l], 1.0/param[l].length);
    		} else {
    			param[l] = Arrays.stream(param[l]).map(x->x/tmp).toArray(Double[]::new);
    		}
    	}
    }
}

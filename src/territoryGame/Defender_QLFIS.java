package territoryGame;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Defender_QLFIS extends Defender {
	
	// actor/controller FLC:
    double U_cal = 0.0; // defuzzified action value from the continuous action space.
    Double[][] q_la; // K_l of FLC
    Double[][] bestDist_q_la;
    int[] chosenActions; // index of chosen action for each rule l in current time step (with epsilon-greedy strategy).
    int[] starActions; // index of best action.
    
    // approximator FIS:
    double Qstar = 0.0; // Q*(t+1)
    double Q = 0.0; // Q(t)
    Double[][] q_lzeta; // K_l of FIS
    Double[][] bestDist_q_lzeta;

    double[] phi_l_u; // membership degree for the approximator FIS based on different input MF parameters.
    List<Double[][]> et_FISParams; // //eligibility trace of FIS: q(l,zeta): M * action.length, FIS mean: M * u.length, FIS sigma: M * u.length
	List<Double[][]> bestDist_et;
    
	// shared:
	List<Double[][]> bestDist_params; // best input params when closest to invader.

	// common params e.g. learning rate, decay factor, etc.
    double alpha = 0.02; // learning rate for approximator FIS.
    double beta = 0.01; // learning rate for controller FLC.
    double sigma = 0.05; // stddev of the gaussian random white noise added to the calculated action u.
	double lambda = 0.9; // eligibility trace factor
		
	Defender_QLFIS(String defenderName, int ALL_DOTS, int b_width, int b_height, double capturedist, int Dx0, int Dy0,
			int cx, int cy, int[] tx, int[] ty, int[] Ix0, int[] Iy0, Double[][] qla, Double[][] qlzeta,
			List<Double[][]> inputParams, List<Double[][]> eligibilityTrace, int[] rulescount) {
		super(defenderName, ALL_DOTS, b_width, b_height, capturedist, Dx0, Dy0, cx, cy, tx, ty, 
				Ix0, Iy0, rulescount, "gaussian");
		
		initMF_QLFIS(inputParams);
		init_qTables(qla, qlzeta);
		initEligibilityTrace(eligibilityTrace);	
	}
	
	private void init_qTables(Double[][] qla, Double[][] qlzeta) {
		actions = createDoubleRange(-Math.PI, Math.PI, 8);
		
	    if (qla[0][0].equals(smallNr)) {
		    q_la = new Double[M][actions.length];
		    q_lzeta = new Double[M][actions.length];
		    for (int l=0; l<M; l++) {
		    	Arrays.fill(q_la[l],1.0/actions.length);
		    	Arrays.fill(q_lzeta[l], 1.0/actions.length);
		    }
	    } else {
	    	q_la = (Double[][])copyArray(Double.class, qla);
	    	q_lzeta = (Double[][])copyArray(Double.class, qlzeta);
	    }
	    bestDist_q_la = (Double[][])copyArray(Double.class,q_la);
	    bestDist_q_lzeta = (Double[][])copyArray(Double.class,q_lzeta);

	    initChosenAction();
	}
	
    private void initChosenAction() {
	    chosenActions = new int[M];
	    starActions = new int[M];
	    phi_l_u = new double[M];
    	Random rand = new Random();
    	for (int l=0; l<M; l++) {
      		int idx = rand.nextInt(actions.length);
       		chosenActions[l] = idx;
    	}
    	
	    previousAngle = Math.abs(Math.PI - Math.abs(inputX[0]-inputX[1]));
	    previousDistToInvader = distToInvader;
	    bestDistToInvader = distToInvader;
	    
    	phi_l = calculatePhi_l(phi_l, inputX, MF_params.subList(0, 2));
    	phi_l_u = calculatePhi_l(phi_l_u, inputX, MF_params.subList(2, 4));
    	calculateController();
    	calculateApproximator(false);
    }
        
	private void initEligibilityTrace(List<Double[][]> eligibilityTrace) {
		et_FISParams = new ArrayList<Double[][]>();
		bestDist_et = new ArrayList<Double[][]>();
		if (eligibilityTrace.get(0)[0][0].equals(smallNr)) {
			et_FISParams.add(new Double[M][actions.length]);
			et_FISParams.add(new Double[M][inputX.length]);
			et_FISParams.add(new Double[M][inputX.length]);
			for (int i=0;i<et_FISParams.size();i++) {
				for (int l=0;l<et_FISParams.get(i).length;l++)
					Arrays.fill(et_FISParams.get(i)[l],0.0);
				bestDist_et.add((Double[][])copyArray(Double.class, et_FISParams.get(i)));
			}
		} else {
			for (int i=0;i<eligibilityTrace.size();i++) {
				et_FISParams.add((Double[][])copyArray(Double.class,eligibilityTrace.get(i)));
				bestDist_et.add((Double[][])copyArray(Double.class,eligibilityTrace.get(i)));
			}
		}
	}
	
    private void initMF_QLFIS(List<Double[][]> inputParams) {
    	initMF(inputParams);
    	bestDist_params = new ArrayList<Double[][]>();
    	if (inputParams.get(0)[0][0].equals(smallNr)) {
	    	MF_params.add((Double[][])copyArray(Double.class,MF_params.get(0)));
	    	MF_params.add((Double[][])copyArray(Double.class,MF_params.get(1)));
    	} 
		for (int i=0;i<MF_params.size();i++) {
			bestDist_params.add(i, (Double[][])copyArray(Double.class, MF_params.get(i)));
		}
	    int len = MFparamsLower.length;
		MFparamsLower = Arrays.copyOf(MFparamsLower, MFparamsLower.length * 2); // mean, stddev
		MFparamsUpper = Arrays.copyOf(MFparamsUpper, MFparamsUpper.length * 2);
	    for (int l=0;l<len;l++) {
	    	MFparamsLower[len+l] = MFparamsLower[l];
	    	MFparamsUpper[len+l] = MFparamsUpper[l];
	    }
    }    
    
	@Override
    protected void calDefenderPos(int dots, int[] Ix0, int[] Iy0) {
    	observe(Ix0, Iy0); // get new inputs.

    	// calculate theoretical best action based on new input:
    	chooseAction(true, dots); // get starActions updated based on new input.
    	phi_l_u = calculatePhi_l(phi_l_u, inputX, MF_params.subList(2, 4));
    	calculateApproximator(true); // calculate Q*_{t+1} for next time step based on starActions    	
    	
    	runOneStep(); // run one step based on U_{t} from previous time step.
    	    	
    	chooseAction(false, dots);
    	phi_l = calculatePhi_l(phi_l, inputX, MF_params.subList(0, 2));
    	calculateController();
    	
    	phi_l_u = calculatePhi_l(phi_l_u, inputX, MF_params.subList(2, 4));
    	calculateApproximator(false); // calculate Q_tp1 for next time step.
    	
    	double err = calculateTDerror();
    	updateElibigilityTrace();
    	updateParams(err); // update FLC's q(l,a), means and devs; FIS's q(l,zeta), means and devs
    	
    	updateEpsilon(dots);
    	updateLearningRate(dots);
    }
	
    /**
     * Choose action for each rule: for epsilon, choose randomly according to uniform distr. 
     * for 1-epsilon, choose the action with max. probability.
     * update arrays of chosen action index and best action index.
     * @param star: if true then update best action index. if false then update chosen action index.
     * @param dots: time step within one round of the game.
     */
    private void chooseAction(boolean star, int dots) {
    	Random rand = new Random();    	
    	for (int l=0; l<M; l++) {
        	double r = rand.nextDouble();
    		if (star) {
    			// for Q*(l,u'): choose from q(l,zeta) the most likely.
    			if (dots>BURNIN) {
    				List<Double> prob = Arrays.stream(q_lzeta[l]).collect(Collectors.toList());
    	        	List<Integer> allMaxes = IntStream.range(0, prob.size()).boxed()
			                .filter(i -> prob.get(i).equals(Collections.max(prob)))
			                .collect(Collectors.toList());
    				// if there are multiple max probabilities, choose randomly among them.
    				starActions[l] = allMaxes.get(rand.nextInt(allMaxes.size()));
    			} else {
    				// if still within burn-in period (e.g. the first 10 timesteps), then choose randomly with uniform distr.
    				starActions[l] = rand.nextInt(actions.length);
    			}
    				
    		} else {
            	if (r<=epsilon[l]) {
            		int idx = rand.nextInt(actions.length);
            		chosenActions[l] = idx;
            	} else {
            		// for choosing FLC actions based on epsilon-greedy: choose from q(l,a) the most likely.
            		List<Double> prob = Arrays.stream(q_la[l]).collect(Collectors.toList());
    	        	List<Integer> allMaxes = IntStream.range(0, prob.size()).boxed()
			                .filter(i -> prob.get(i).equals(Collections.max(prob)))
			                .collect(Collectors.toList());
        			chosenActions[l] = allMaxes.get(rand.nextInt(allMaxes.size()));
            	}
    		}
    	}
    }
	
	private void calculateController() {
    	U_cal = 0.0;
    	for (int l=0; l<M; l++) {
    		U_cal = U_cal + phi_l[l] * actions[chosenActions[l]]; // different from Q-learning
    	}
    	Random rand = new Random();
    	double r = rand.nextGaussian() * sigma;
    	U = U_cal + r;
	}
	
    /**
     * @param star: if true then calculate and update Q*. if false then calculate and update Q.
     */
    private void calculateApproximator(boolean star) {
    	if (star) {
    		Qstar = 0.0;
    		for (int l=0; l<M; l++)
    			Qstar = Qstar + phi_l_u[l] * q_lzeta[l][starActions[l]];
    	} else {
    		Q = 0.0;
    		for (int l=0; l<M; l++){
    			Q = Q + phi_l_u[l] * q_lzeta[l][chosenActions[l]];
	        }
    	}
    }
	
    /**
     * calculate partial derivative for gaussian MF as inputs; update input params accordingly.
     * @param input: for FLC it's inputX. for FIS it's U.
     * @param output_discrete: K_l. for FLC it's actions, for FIS it's zeta.
     * @param output_continuous: u or Q. u is the output of the actor, Q that of the critic.
     * @param phi: corresponding phi(l) value for FLC or FIS. phi_l for FLC, phi_l_u for FIS.
     * @param l: index for rules.
     * @param i: index for inputs.
     * @param lr: learning rate. alpha for FIS, beta for FLC.
     * @param params: MF_params.subList(0,2) for FLC_mean, FLC_stddev; MF_params.sublist(2,4) for FIS_mean, FIS_stddev. 
     * @return delta mean and delta sigma values.
     */
    private double[] getGaussianUpdate(double[] input, double output_discrete_l, double output_continuous, double[] phi, 
    									int l, int i, double lr, List<Double[][]> params) {
    	double delta_sigma = (output_discrete_l - output_continuous) * phi[l] * 2 * Math.pow((input[i] - params.get(0)[l][i]),2) / Math.pow((params.get(1)[l][i]),3);
    	double delta_mean = (output_discrete_l - output_continuous) * phi[l] * 2 * (input[i] - params.get(0)[l][i]) / Math.pow((params.get(1)[l][i]),2);
    	
    	return new double[]{delta_mean, delta_sigma};
    }
    
    private void updateElibigilityTrace() {
    	// K_l / q(l,zeta):
    	for (int l=0; l<M; l++) {
    		et_FISParams.get(0)[l][chosenActions[l]] = et_FISParams.get(0)[l][chosenActions[l]] + phi_l_u[l];
    		if (et_FISParams.get(0)[l][chosenActions[l]]<=ZERO)
    			et_FISParams.get(0)[l][chosenActions[l]] = 0.0;
    	}    	
    	for (int l=0;l<et_FISParams.get(1).length;l++) {
    		for (int i=0; i<et_FISParams.get(1)[0].length;i++) {
				double[] delta = getGaussianUpdate(inputX, q_lzeta[l][chosenActions[l]], Q, phi_l_u, 
						l, i, alpha, MF_params.subList(2, 4));
				// mean:
				et_FISParams.get(1)[l][i] = gamma * lambda * et_FISParams.get(1)[l][i] + delta[0];
				if (Math.abs(et_FISParams.get(1)[l][i]) <= ZERO)
					et_FISParams.get(1)[l][i] = 0.0;
				// sigma:
				et_FISParams.get(2)[l][i] = gamma * lambda * et_FISParams.get(2)[l][i] + delta[1];
				if (Math.abs(et_FISParams.get(2)[l][i]) <= ZERO)
					et_FISParams.get(2)[l][i] = 0.0;
    		}
    	}    	
    }
    
    private double calculateTDerror() {
    	double reward = calculateRewardShaping();
    	updateBestParams_QLFIS();
    	return reward + gamma * Qstar - Q;
    }
    
    private void updateParams(double TDerror) {
    	// FLC:
    	updateFLCOutput(TDerror);
    	updateFLCInput(MF_params.get(0), MFparamsLower[0], MFparamsUpper[0], TDerror);
    	updateFLCInput(MF_params.get(1), MFparamsLower[1], MFparamsUpper[1], TDerror);
    	
    	// FIS:
    	updateFISOutput(TDerror, et_FISParams.get(0));
    	updateFISInput(MF_params.get(2), MFparamsLower[2], MFparamsUpper[2], TDerror, et_FISParams.get(1));
    	updateFISInput(MF_params.get(3), MFparamsLower[3], MFparamsUpper[3], TDerror, et_FISParams.get(2));
    }
    
    private void updateFLCInput(Double[][] param, double paramLower, double paramUpper, double TDerror) {
    	for (int l=0;l<param.length;l++) {
    		final int i = l;
    		param[i] = Arrays.stream(param[i]).map(x -> x + beta * TDerror * phi_l[i] * (U-U_cal) / sigma).toArray(Double[]::new);
    		for (int j=0;j<param[i].length;j++) {
    			param[i][j] = Math.max(param[i][j], paramLower);
    			param[i][j] = Math.min(param[i][j], paramUpper);
    		}
    	}
    }
    
    private void updateFLCOutput(double TDerror) {
    	// K_l / q(l,a):
    	for (int l=0; l<M; l++) {
    		q_la[l][chosenActions[l]] = q_lzeta[l][chosenActions[l]] + beta * TDerror * phi_l[l] * (U-U_cal) / sigma;
    		q_la[l][chosenActions[l]] = q_la[l][chosenActions[l]] < ZERO ? 0 : q_la[l][chosenActions[l]];
    	}
    	
//    	if (name.equals("QLFIS_d0")) {
//			System.out.print("beta: "+beta+" TD: "+TDerror+" U: "+U+" Ucal: "+U_cal+" sigma: "+sigma+" \nq(l=5,a): ");
//	    	printArray(Arrays.stream(q_la[5]).collect(Collectors.toList()), " ");
//	    	printArray(Arrays.stream(phi_l).boxed().collect(Collectors.toList()), "phi_l: ");
//    		printArray(IntStream.of(chosenActions).boxed().collect(Collectors.toList()), "chosen: ");
//    		printArray(IntStream.of(starActions).boxed().collect(Collectors.toList()), "star  : ");
//    		System.out.print("incremental: ");
//    		for (int l=0;l<M;l++)
//    			System.out.print((beta * TDerror * phi_l[l] * (U-U_cal) / sigma) + " ");
//    		System.out.print("\n");
//    	}

    	
    	normalizeRows(q_la);
    }
    
    private void updateFISInput(Double[][] param, double paramLower, double paramUpper, double TDerror, Double[][] et) {
    	for (int l=0;l<param.length;l++) {
    		final int i = l;
    		param[i] = IntStream.range(0, param[i].length).mapToDouble(idx -> param[i][idx] + alpha * TDerror * et[i][idx]).boxed().toArray(Double[]::new);
    		for (int j=0;j<param[i].length;j++) {    			
    			param[i][j] = Math.max(param[i][j], paramLower);
    			param[i][j] = Math.min(param[i][j], paramUpper);
    		}
    	}
    }
    
    private void updateFISOutput(double TDerror, Double[][] et) {
    	// K_l / q(l,zeta):
    	for (int l=0; l<M; l++) {
    		q_lzeta[l][chosenActions[l]] = q_lzeta[l][chosenActions[l]] + alpha * TDerror * et[l][chosenActions[l]];
    		q_lzeta[l][chosenActions[l]] = q_lzeta[l][chosenActions[l]] < ZERO ? 0 : q_lzeta[l][chosenActions[l]];
    	}
    	normalizeRows(q_lzeta);
    }
    
    private void updateBestParams_QLFIS() {
    	if (distToInvader <= bestDistToInvader) {
    		// update params until the most recent catch. therefore the "<=" instead of "<"
    		updateBestParams_basic();
    		for(int i=0;i<et_FISParams.size();i++) {
    			bestDist_et.set(i, (Double[][])copyArray(Double.class,et_FISParams.get(i)));
    		}
    		for (int i=0;i<MF_params.size();i++) {
    			bestDist_params.set(i, (Double[][])copyArray(Double.class, MF_params.get(i)));
    		}
    		bestDist_q_la = (Double[][])copyArray(Double.class,q_la);
    		bestDist_q_lzeta = (Double[][])copyArray(Double.class,q_lzeta);
    	}
    }
    
    /**
     * update alpha and beta value (decrease learning rate).
     * @param dots
     */
    private void updateLearningRate(int dots) {
//    	alpha = updateParam(alpha, dots);
//    	beta = updateParam(beta,dots);
    	
    }
}

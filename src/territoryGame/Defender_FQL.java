package territoryGame;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Defender_FQL extends Defender {
    
    Double[][] q_la; // q(l,a) is the probability of choosing action a_i for each rule. dim=M*Actions.length.
    Double[][] bestDist_q_la; // q(l,a) at closest distance to invader.
    
    int[] chosenActions; // index of chosen action for each rule l in current time step (with epsilon-greedy strategy).
    int[] starActions; // index of best action for each rule l in t+1 time step (for max q(l,a)).

    double Qstar = 0.0; // Q*(t+1)
    double Q = 0.0; // Q(t)

    double eta = 0.2; // learning rate for updating coefs.
    
	Defender_FQL(String defenderName, int ALL_DOTS, int b_width, int b_height, double capturedist, 
			int Dx0, int Dy0, int cx, int cy, int[] tx, int[] ty, 
			int[] Ix0, int[] Iy0, Double[][] qla, int[] rulescount, String mf){
		
		super(defenderName, ALL_DOTS, b_width, b_height, capturedist, 
				Dx0, Dy0, cx, cy, tx, ty, Ix0, Iy0, rulescount, mf);
		
	    initGameFQL(qla);
	}
	
	/**
	 * 
	 * @param qla: initial q-table.
	 */
	private void initGameFQL(Double[][] qla) {
		
	    actions = createDoubleRange(-Math.PI, Math.PI, 8); // by default 8 discrete actions, ranged from -pi to pi.
	    if (qla[0][0].equals(smallNr)) {
		    q_la = new Double[M][actions.length];
		    for (int l=0; l<M; l++)
		    	Arrays.fill(q_la[l],1.0/actions.length);	    	
	    } else {
	    	q_la = (Double[][])copyArray(Double.class,qla);
	    }
	    bestDist_q_la = (Double[][])copyArray(Double.class,q_la);

	    initChosenAction();
    	
	    previousAngle = Math.abs(Math.PI - Math.abs(inputX[0]-inputX[1]));
	    previousDistToInvader = distToInvader;
	    bestDistToInvader = distToInvader;
    	phi_l = calculatePhi_l(phi_l, inputX, MF_params);
    	calculateU();
    	calculateQ(false);
	}
	
    private void initChosenAction() {
	    chosenActions = new int[M];
	    starActions = new int[M];
    	Random rand = new Random();
    	for (int l=0; l<M; l++) {
      		int idx = rand.nextInt(actions.length);
      		chosenActions[l] = idx;
      		starActions[l] = idx;
    	}
    }
    
    @Override
    protected void calDefenderPos(int dots, int[] Ix0, int[] Iy0) {
    	observe(Ix0, Iy0); // get new inputs.
    	phi_l = calculatePhi_l(phi_l, inputX, MF_params); // update membership degree based on new inputs.
    	chooseAction(true, dots); // calculate best action based on new inputs.
    	calculateQ(true); // calculate Q*(t) based on new inputs.
    	 		
    	runOneStep();
    	
    	updateQla(); // use Q(t-1), Q*(t), reward(t), Phi_l(t) to update q(l,a) table for choosing new action.
    	chooseAction(false, dots); // choose action with epsilon-greedy strategy based on updated prob in q(l,a).
    	calculateU();
    	calculateQ(false); // calculate Q(t) for next time step.
    	updateEpsilon(dots);
    	eta = updateParam(eta, dots);
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
    		List<Double> prob = Arrays.stream(q_la[l]).collect(Collectors.toList());
			List<Integer> allMaxes = IntStream.range(0, prob.size()).boxed()
							                .filter(i -> prob.get(i).equals(Collections.max(prob)))
							                .collect(Collectors.toList());
    		if (star) {
    			if (dots>BURNIN) {
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
        			chosenActions[l] = allMaxes.get(rand.nextInt(allMaxes.size()));
            	}
    		}
    	}
//    	if(!star)
//    		printArray(IntStream.of(chosenActions).boxed().collect(Collectors.toList()), name+":");
    }
    
    private void calculateU() {
    	U = 0.0;
//    	printArray(DoubleStream.of(actions).boxed().collect(Collectors.toList()),name+":");
    	for (int l=0; l<M; l++) {
    		U = U + phi_l[l] * actions[chosenActions[l]];
    	}
    }

    /**
     * @param star: if true then calculate and update Q*. if false then calculate and update Q.
     */
    private void calculateQ(boolean star) {
    	if (star) {
    		Qstar = 0.0;
    		for (int l=0; l<M; l++)
    			Qstar = Qstar + phi_l[l] * q_la[l][starActions[l]];
    	} else {
    		Q = 0.0;
    		for (int l=0; l<M; l++){
    			Q = Q + phi_l[l] * q_la[l][chosenActions[l]];
	        	}
    	}
    }
    
    private void updateBestParams() {   	    	
    	if (distToInvader <= bestDistToInvader) {
    		updateBestParams_basic();
    		bestDist_q_la = (Double[][])copyArray(Double.class,q_la);
    	}
    }
    
    private double calculateTDerror() {
    	double reward = calculateRewardShaping();
    	updateBestParams();
    	return reward + gamma * Qstar - Q;
    }
    
    protected void updateQla() {
    	double TDerror = calculateTDerror();
    	for (int l=0; l<M; l++) {
    		q_la[l][chosenActions[l]] = q_la[l][chosenActions[l]] + eta * TDerror * phi_l[l];
    		q_la[l][chosenActions[l]] = q_la[l][chosenActions[l]] < 0 ? 0 : q_la[l][chosenActions[l]];
    	}
    	normalizeRows(q_la);
    }
    

}

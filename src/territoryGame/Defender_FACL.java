package territoryGame;

import java.util.Arrays;
import java.util.Random;

public class Defender_FACL extends Defender{
   
    double[] zeta; // corresponding value of the actions (output of the critic). length=number of rules
	double[] bestActions;
	double[] bestZeta;
    
    double U_cal = 0.0; // defuzzified action value from the continuous action space.
    double V_tm1 = 0.0; // V(t-1)
    double V = 0.0; // V(t)
    
    double alpha = 0.02; // learning rate for critic.
    double beta = 0.01; // learning rate for actor.
    double sigma = 0.05; // stddev of the gaussian random white noise added to the calculated action u.
    
	Defender_FACL(String defenderName, int ALL_DOTS, int b_width, int b_height, double capturedist, 
			int Dx0, int Dy0, int cx, int cy, int[] tx, int[] ty, int[] Ix0, int[] Iy0, 
			double[] actorInit, double[] criticInit, int[] rulescount, String mf){

		super(defenderName, ALL_DOTS, b_width, b_height, capturedist, 
				Dx0, Dy0, cx, cy, tx, ty, Ix0, Iy0, rulescount, mf);
		
	    initGame_FACL(actorInit, criticInit);
	}
	
	/**
	 * @param actorInit: initial actor output params
	 * @param criticInit: initial critic output params
	 */
	private void initGame_FACL(double[] actorInit, double[] criticInit) {
	    
	    Random rand = new Random();
	    double ttlInit = Arrays.stream(actorInit).sum();
	    if (ttlInit==smallNr) {
	    	// round 0
	    	actions = rand.doubles(M, -Math.PI, Math.PI).toArray();
	    	zeta = rand.doubles(M, -Math.PI, Math.PI).toArray();
	    } else {
	    	actions = actorInit.clone();
	    	zeta = criticInit.clone();
	    }
	    
	    bestActions = actions.clone();
	    bestZeta = zeta.clone();
    	
	    previousAngle = Math.abs(Math.PI - Math.abs(inputX[0]-inputX[1]));
	    previousDistToInvader = distToInvader;
	    bestDistToInvader = distToInvader;
    	phi_l = calculatePhi_l(phi_l, inputX, MF_params);
    	calculateActor();
    	calculateCritic();
	}
	
	@Override
    protected void calDefenderPos(int dots, int[] Ix0, int[] Iy0) {
    	observe(Ix0, Iy0); // get new inputs.
    	phi_l = calculatePhi_l(phi_l, inputX, MF_params); // update membership degree based on new inputs.
    	calculateActor(); // find u for next move.
    	calculateCritic(); // calculate Q_tp1 for next time step.
    	
    	runOneStep();
    	    	
    	updateOutput(); // calculate reward, update actions and zetas.
    	updateEpsilon(dots);
    	updateLearningRate(dots);
    }    
    
    /**
     * output: U_cal: calculated action. U: actual action with random factor.
     * https://pdfs.semanticscholar.org/cf89/b9ea56aede40dc5191b091d9304fd125287e.pdf
     */
    private void calculateActor() {
    	U_cal = 0.0;
//    	printArray(DoubleStream.of(actions).boxed().collect(Collectors.toList()));
    	for (int l=0; l<M; l++) {
    		U_cal = U_cal + phi_l[l] * actions[l]; // different from Q-learning
    	}

    	Random rand = new Random();
    	double r = rand.nextGaussian() * sigma;
    	U = U_cal + r;
    }

    private void calculateCritic() {
    	V_tm1 = V;
		V = 0.0;
		for (int l=0; l<M; l++){
			V = V + phi_l[l] * zeta[l];
        }
    }
    
    private void updateBestParams() {
    	if (distToInvader <= bestDistToInvader) {
    		updateBestParams_basic();
    		bestActions = actions.clone();
    		bestZeta = zeta.clone();
    	}
    }
    
    private double calculateTDerror() {
    	double reward = calculateRewardShaping();
    	updateBestParams();
    	return reward + gamma * V - V_tm1;
    }
    
    /**
     * update output of actor (actions) and critic (zeta).
     */
    private void updateOutput() {
    	
    	double TDerror = calculateTDerror();
    	for (int l=0;l<M;l++) {
    		zeta[l] = zeta[l] + TDerror * alpha * phi_l[l];
    	}
    	
    	for (int l=0;l<M;l++) {
    		actions[l] = actions[l] + Math.signum(TDerror * (U-U_cal) / sigma ) * beta * phi_l[l];
    		if ( (actions[l] >= Math.PI) || (actions[l] < -Math.PI) )
    			actions[l] = -Math.PI;
    	}
    }
    
    /**
     * update alpha and beta value (decrease learning rate).
     * @param dots
     */
    private void updateLearningRate(int dots) {
    	alpha = updateParam(alpha, dots);
    	beta = updateParam(beta,dots);
    }
}

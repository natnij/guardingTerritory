package territoryGame;

import java.awt.EventQueue;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import javax.swing.JFrame;

public class Territory extends JFrame {

	/**
	 * 
	 */
	private static final long serialVersionUID = 584836517988920804L;
	GameBoard gb;
	String gameType; // FQL for fuzzy Q-learning, FACL for fuzzy actor-critic, or QLFIS for Q(lambda)-learning fuzzy inference system
	String MF = "gaussian"; // membership function can be gaussian or triangular.
	private int count;
	private int gameCount = 0;
	private PropertyChangeListener pcl;
	private Double[][] qla; // q(l,a) params for FQL and QLFIS
	private double[] actor; // actor params for FACL
	private double[] critic; // critic params for FACL
	private Double[][] qlzeta; // q(l,zeta) params for QLFIS
	private List<Double[][]> et; // eligibility traces for QLFIS
	private List<Double[][]> MF_params; // input MF params for QLFIS
	private int[] updateRulesCount;
	private int nrRounds = 100;
	private int nrGames = 20;
	private int showGameInterval = 10;
	private List<Integer> gamesWon = new ArrayList<Integer>();
	private List<Double> distInvaderToTerritory;
	private Logger logger;
	private final double smallNr = -1E6;
	private final double bigNr = 1E6;
	
    public Territory(String type) {
    	initLog();
    	initParams(type);
    	pcl = new PropertyChangeListener() {
			@Override
			public void propertyChange(PropertyChangeEvent evt) {
				restart();
				count ++;
			}
    	};
        initUI();
    }
    
    private void restartExec() {
    	remove(gb);
    	setupGameBoard();
        revalidate();
        gb.requestFocusInWindow();
    }
    
    private void restart() {
    	if (gameCount < nrGames) {   	
	        if (count < nrRounds) {
	        	collectData();
	        	restartExec();
	        } else {
	        	collectData();
	        	gameCount++;
	        	initParams(gameType);
	        	restartExec();
	        }
    	} else {
    		gb.removePropertyChangeListener(pcl);
    	}
    }
    
    private void initUI() {
    	
    	setupGameBoard();
        
        // start the UI
        setResizable(false);
        pack();
        
        setTitle("Territory"+gameType);
        setLocationRelativeTo(null);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
    
    private void initParams(String type) {
    	gameType = type;
    	count = 0;
    	distInvaderToTerritory = new ArrayList<Double>();
    	qla = new Double[][]{{smallNr}};
    	actor = new double[]{smallNr};
    	critic = new double[]{smallNr};
    	qlzeta = new Double[][]{{smallNr}};
    	et = new ArrayList<Double[][]>();
    	et.add(new Double[][]{{smallNr}});
    	MF_params = new ArrayList<Double[][]>();
    	MF_params.add(new Double[][]{{smallNr}});
    	updateRulesCount = new int[]{0};
    }
    
    private void setupGameBoard() {
    	if (gameType.equals("FQL")) {
    		gb = new GameBoard(qla, updateRulesCount, count, MF); 
    	} else if (gameType.equals("FACL")) {
    		gb = new GameBoard(actor, critic, updateRulesCount, count, MF);
    	} else if (gameType.equals("QLFIS")) {
    		gb = new GameBoard(qla, qlzeta, MF_params, et, updateRulesCount, count);
    	}
    	gb.addPropertyChangeListener(pcl);
        add(gb);
    }
    
    private void collectData() {
    	if (gameType.equals("FQL")) {
    		collectFQLData();
    	} else if (gameType.equals("FACL")) {
    		collectFACLData();
    	} else if (gameType.equals("QLFIS")) {
    		collectQLFISData();
    	}
    	collectUpdateRulesCount();
    	showGamesWon();
    }
    
    private void updateMatrix(Double[][] mat, double weight, Double[][] updateMat) {
    	// https://docs.oracle.com/javase/8/docs/api/java/util/stream/Stream.html#toArray-java.util.function.IntFunction-
		final int a = mat[0].length;
    	for(int l=0; l<mat.length; l++){
    		final int i = l;
    		mat[i] = IntStream.range(0,a)
    				.mapToDouble(idx -> mat[i][idx] + weight * updateMat[i][idx]).boxed().toArray(Double[]::new);
    	}
    }
    
    private double[] updateMatrix(double[] vec, double weight, double[] updateVec) {
    	final int m = vec.length;
    	final double[] vecCopy = vec.clone();
    	vec = IntStream.range(0, m).mapToDouble(idx -> vecCopy[idx] + weight * updateVec[idx]).toArray();
    	return vec;
    }
    
    private void normalizeRows(Double[][] param) {
    	for (int l=0; l<param.length; l++) {
    		Double tmp = Arrays.stream(param[l]).mapToDouble(Double::doubleValue).sum();
    		if (tmp==0) {
    			Arrays.fill(param[l], 1.0/param[l].length);
    		} else {
    			param[l] = Arrays.stream(param[l]).map(x->x/tmp).toArray(Double[]::new);
    		}
    	}
    }
    
    private void normalizeElements(Double[][] param, double[] weight) {
    	double w = Arrays.stream(weight).sum();
    	for (int l=0;l<param.length;l++) {
    		param[l] = Arrays.stream(param[l]).map(x->x/w).toArray(Double[]::new);
    	}
    }
    private void getDistInvaderToTerritory() {
    	for (int i=0;i<gb.nrInvaders;i++) {
    		distInvaderToTerritory.add(gb.invaders.get(i).distToTerritory - gb.R);
    	}
    }
    
    private void collectFQLData() {
    	if (qla[0][0].equals(smallNr)) {
    		int M = gb.defenders.get(0).M;
  			int A = ((Defender_FQL)gb.defenders.get(0)).actions.length;
    		qla = new Double[M][A];
		    for (int l=0; l<M; l++)
		    	Arrays.fill(qla[l],1.0/A);
    	}
    	for (int d=0;d<gb.nrDefenders;d++) {
    		Defender_FQL defender = (Defender_FQL) gb.defenders.get(d);
    		double weight = defender.winner ? (defender.distInvaderToTerritory) : 1.0 / defender.bestDistToInvader;
    		updateMatrix(qla, weight, defender.bestDist_q_la);
    	}
    	normalizeRows(qla);
    	getDistInvaderToTerritory();
    }
    
    private void collectFACLData() {
    	if(Arrays.stream(actor).sum()==smallNr) {
    		int M = gb.defenders.get(0).M;
  			actor = new double[M];
  			critic = new double[M];
    	}
		Arrays.fill(actor, 0.0);
		Arrays.fill(critic, 0.0);
    	
    	double[] weight = new double[gb.nrDefenders];
    	for (int d=0;d<gb.nrDefenders;d++) {
    		Defender_FACL defender = (Defender_FACL) gb.defenders.get(d);
    		weight[d] = defender.winner ? (defender.distInvaderToTerritory) : 1.0 / defender.bestDistToInvader;
    	}
    	// normalize weight
    	final double ttlWeight = Arrays.stream(weight).sum();
    	weight = Arrays.stream(weight).map(x -> x / ttlWeight).toArray();
    	
    	for (int d=0;d<gb.nrDefenders;d++) {
    		Defender_FACL defender = (Defender_FACL) gb.defenders.get(d);
        	final double w = weight[d];
        	actor = updateMatrix(actor, w, defender.bestActions);
    		critic = updateMatrix(critic, w, defender.bestZeta);
    	}
		actor = Arrays.stream(actor).map(x -> ((x >= Math.PI)||(x<-Math.PI)) ? -Math.PI : x).toArray();
		getDistInvaderToTerritory();
    }
    
    private void collectQLFISData() {
		int M = gb.defenders.get(0).M;
		int A = ((Defender_QLFIS)gb.defenders.get(0)).actions.length;
		int X = ((Defender_QLFIS)gb.defenders.get(0)).inputX.length;
    	
		if (qla[0][0].equals(smallNr)) {
    		qla = new Double[M][A];
    		qlzeta = new Double[M][A];
		    for (int l=0; l<M; l++) {
		    	Arrays.fill(qla[l],1.0/A);
		    	Arrays.fill(qlzeta[l],1.0/A);
		    }
    	}
    	
	    et = new ArrayList<Double[][]>();
		et.add(new Double[M][A]);
		et.add(new Double[M][X]);
		et.add(new Double[M][X]);
		for (int i=0;i<et.size();i++) {
			for (int l=0;l<et.get(i).length;l++)
				Arrays.fill(et.get(i)[l],0.0);
		}
		MF_params = new ArrayList<Double[][]>();
		for (int i=0;i<4;i++)
			MF_params.add(new Double[M][X]);
		for (int i=0;i<MF_params.size();i++) {
			for (int l=0;l<MF_params.get(i).length;l++)
				Arrays.fill(MF_params.get(i)[l], 0.0);
		}
		
    	double[] weight = new double[gb.nrDefenders];
    	for (int d=0;d<gb.nrDefenders;d++) {
    		Defender_QLFIS defender = (Defender_QLFIS) gb.defenders.get(d);
    		weight[d] = defender.winner ? (defender.distInvaderToTerritory) : 1.0 / defender.bestDistToInvader;
    		updateMatrix(qla, weight[d], defender.bestDist_q_la);
    		updateMatrix(qlzeta, weight[d], defender.bestDist_q_lzeta);

    		for (int i=0;i<et.size();i++) {
    			updateMatrix(et.get(i), weight[d], defender.bestDist_et.get(i));
    		}

//    		print2DArray(MF_params.get(0), 0, "FLC MF mean before "+defender.name);
    		
    		for (int i=0;i<MF_params.size();i++) {
    			updateMatrix(MF_params.get(i), weight[d], defender.bestDist_params.get(i));
    		}
    		
//    		print2DArray(MF_params.get(0), 0, "FLC MF mean after "+defender.name);
    	}
    	normalizeRows(qla);
    	normalizeRows(qlzeta);
    	for (int i=0;i<MF_params.size();i++) {
    		normalizeElements(MF_params.get(i), weight);
    	}
//    	print2DArray(MF_params.get(0), 0, "FLC MF mean after normalizing");
    	
    	getDistInvaderToTerritory();
    }
    
    private void collectUpdateRulesCount() {
    	int M = gb.defenders.get(0).M;
    	if(Arrays.stream(updateRulesCount).sum()==0) {
    		updateRulesCount = new int[M];
    	}
    	for (int d=0;d<gb.nrDefenders;d++) {
    		Defender defender = gb.defenders.get(d);
    		// scale down by factor of 10
    		updateRulesCount = IntStream.range(0, M).map(idx -> Math.min(updateRulesCount[idx] + defender.bestUpdateRulesCount[idx] / 10, (int)bigNr)).toArray();
    	}
    }
    
    private void showGamesWon() {
    	gamesWon.add(gb.defenderWins);
    	if (gamesWon.size() % showGameInterval == 0) {
    		double ct = distInvaderToTerritory.stream().filter(x->x>0).count();
    		double avgDist = ct==0 ? 0 : (distInvaderToTerritory.stream().filter(x->x>0).mapToDouble(x->x).sum() / ct);
    		String msg = gameType+" game "+gameCount+" round "+ count+" won: "+IntStream.range(1, showGameInterval+1).map(idx->gamesWon.get(gamesWon.size()-idx)).sum()+" in "+showGameInterval+" with dist "+avgDist;
    		logger.info(msg);
    		System.out.println(msg);
    		printArray(distInvaderToTerritory, "dist:");
    		distInvaderToTerritory = new ArrayList<Double>();
    	}
    }
    
    private void initLog() {
        logger = Logger.getLogger("MyLog");
        FileHandler fh;  

        try {
            fh = new FileHandler("gameLog.log");  
            logger.addHandler(fh);
            logger.setUseParentHandlers(false);
            MyFormatter formatter = new MyFormatter();  
            fh.setFormatter(formatter);

        } catch (SecurityException e) {  
            e.printStackTrace();  
        } catch (IOException e) {  
            e.printStackTrace();  
        } 
    }
        
    public void print2DArray(Double[][] arr, String remark) {
    	System.out.println(remark + ": ");
    	for (int l=0;l<arr.length;l++)
    		printArray(Arrays.stream(arr[l]).collect(Collectors.toList())," ");
    	System.out.println("\n");
    }
    
    public void print2DArray(Double[][] arr, int column, String remark) {
    	System.out.println(remark + ": ");
    	for (int l=0;l<arr.length;l++)
    		System.out.print(arr[l][column]+" ");
    	System.out.println("\n");
    }
    
    private <T> void printArray(List<T> al, String remark) {
    	System.out.print(remark + " ");
    	for(int i=0;i<al.size();i++)
    		System.out.print(al.get(i) + " ");
    	System.out.print("\n");
    }
    
	public static void main(String[] args)
	{
        EventQueue.invokeLater(() -> {
            JFrame ex0 = new Territory("FACL");
            ex0.setVisible(true);
            
            JFrame ex1 = new Territory("FQL");
            ex1.setVisible(true);
            
            JFrame ex2 = new Territory("QLFIS");
            ex2.setVisible(true);
        });
	}
}

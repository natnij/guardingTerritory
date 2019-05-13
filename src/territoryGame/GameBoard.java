package territoryGame;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;

import javax.swing.ImageIcon;
import javax.swing.JPanel;
import javax.swing.Timer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class GameBoard extends JPanel implements ActionListener {

    /**
	 * 
	 */
	private static final long serialVersionUID = -829755398425341644L;
	private final int B_WIDTH = 300;
    private final int B_HEIGHT = 300;
    private final int ALL_DOTS = 1800;
    private final int DOT_SIZE = 10;
    private final int DELAY = 1;
    private final double CAPTUREDIST = 15.0; 
    final int R = 50;
    
    private int dots;
    private final int Cx = B_WIDTH - 3*R;
    private final int Cy = B_HEIGHT - R;
    private final int Tx[] = new int[360];
    private final int Ty[] = new int[360];
    
    int nrDefenders = 2;
    int nrInvaders = 1;
    int nrInvadersInGame = nrInvaders;
    int nrInvadersInTerritory = 0;
    int defenderWins = 0;
    // length of the vector is equal number of defenders. 
    // each element defines lower limit of the corresponding defender's position on x-axis.
    private final int[] DxRangeL = {10,10,10};
    // each element defines higher limit of the corresponding defender's position on x-axis.    
    private final int[] DxRangeH = {290,290,290};
    private final int[] DyRangeL = {10,10,10};
    private final int[] DyRangeH = {290,290,290};
    private final int[] IxRangeL = {50,250};
    private final int[] IxRangeH = {80,280};
    private final int[] IyRangeL = {10,10};
    private final int[] IyRangeH = {30,30};
    List<Invader> invaders;
    List<Defender> defenders;
    
    private Timer timer;
    private Image invaderTrack;
    private Image invaderNow;
    private Image defenderTrack;
    private Image defenderNow;
    private Image defenderRange;
    private Image territory;
    private Image territoryCenter;
    private Image intersect;
    private Image nearestBorder;
    private boolean inGame = true;
    private boolean endGame = false;
    private TAdapter ka;
    private int BURNIN = 50;
    
    // https://stackoverflow.com/questions/13777936/java-property-change-listener
    private PropertyChangeSupport pcs = new PropertyChangeSupport(this);
    
    public void addPropertyChangeListener(PropertyChangeListener listener) {
    	pcs.addPropertyChangeListener(listener);
    }
    
    public void removePropertyChangeListener(PropertyChangeListener listener) {
    	pcs.removePropertyChangeListener(listener);
    }
    
    public void setEndGame(boolean isEnd) {
    	endGame = isEnd;
    	if (isEnd == true) {
    		removeKeyListener(ka);
    		pcs.firePropertyChange("endGame", false, true);
    	}
    }
    
    public boolean getEndGame() {
    	return endGame;
    }
    
    public GameBoard(Double[][] qla,  int[] rulescount, int count, String mf) {
    	initBoard();
        Coordinates coord = initGame();
        addDefenders(qla, rulescount, count, coord, mf);
        startTimer();
    }
    
    public GameBoard(double[] actor, double[] critic, int[] rulescount, int count, String mf) {
    	initBoard();
    	Coordinates coord = initGame();
    	addDefenders(actor, critic, rulescount, count, coord, mf);
    	startTimer();
    }
    
    public GameBoard(Double[][] qla, Double[][] qlzeta, List<Double[][]> inputParams, 
    		List<Double[][]> eligibilityTrace, int[] rulescount, int count) {
    	initBoard();
    	Coordinates coord = initGame();
    	addDefenders(qla, qlzeta, inputParams, eligibilityTrace, rulescount, count, coord);
    	startTimer();
    }
    
    private void initBoard() {
    	if (nrDefenders<2) 
    		nrDefenders = 2;
    	ka = new TAdapter();
        addKeyListener(ka);
        setBackground(Color.white);
        setFocusable(true);
        setPreferredSize(new Dimension(B_WIDTH, B_HEIGHT));
        loadImages();
    }

    private void loadImages() {

        ImageIcon invaderIcon = new ImageIcon("src/img/invaderTrack.png");
        invaderTrack = invaderIcon.getImage();
        
        ImageIcon invaderNowIcon = new ImageIcon("src/img/invaderPos.png");
        invaderNow = invaderNowIcon.getImage();

        ImageIcon defenderIcon = new ImageIcon("src/img/defenderTrack.png");
        defenderTrack = defenderIcon.getImage();
        
        ImageIcon defenderNowIcon = new ImageIcon("src/img/defenderPos.png");
        defenderNow = defenderNowIcon.getImage();
        
        ImageIcon defenderRangeIcon = new ImageIcon("src/img/defenderRange.png");
        defenderRange = defenderRangeIcon.getImage();
        
        ImageIcon territoryIcon = new ImageIcon("src/img/territory.png");
        territory = territoryIcon.getImage();
        
        ImageIcon territoryCenterIcon = new ImageIcon("src/img/territory.png");
        territoryCenter = territoryCenterIcon.getImage();
        
        ImageIcon interceptIcon = new ImageIcon("src/img/dotO.png");
        intersect = interceptIcon.getImage();
        
        nearestBorder = interceptIcon.getImage();
    }
    
    private Coordinates initGame() {
    	setEndGame(false);
    	locateTerritory();
    	
    	int[] Dx = new int[nrDefenders];
        int[] Dy = new int[nrDefenders];
        int[] Ix = new int[nrInvaders];
        int[] Iy = new int[nrInvaders];
    	Random rand = new Random();
        for (int i=0;i<nrDefenders;i++) {
        	Dx[i] = rand.ints(1, DxRangeL[i], DxRangeH[i]).sum();
        	Dy[i] = rand.ints(1, DyRangeL[i], DyRangeH[i]).sum();
        }
        
        for (int i=0;i<nrInvaders;i++) {
        	Ix[i] = rand.ints(1, IxRangeL[i], IxRangeH[i]).sum();
        	Iy[i] = rand.ints(1, IyRangeL[i], IyRangeH[i]).sum();	
        }
        Coordinates coord = new Coordinates(Dx,Dy,Ix,Iy);
        
        invaders = new ArrayList<Invader>();
        defenders = new ArrayList<Defender>();
        
        addInvaders(coord);
        
        return coord;
    }
    
    private void addInvaders(Coordinates coord) {
        for (int i=0;i<nrInvaders;i++) {
    		invaders.add(new Invader("i"+i, ALL_DOTS, B_WIDTH, B_HEIGHT, 
    				coord.Ix[i], coord.Iy[i], Cx, Cy, Tx, Ty, coord.Dx, coord.Dy));
    	}
    }
    
    private void startTimer() {
        dots = 1;
        timer = new Timer(DELAY, this);
        timer.start();
    }
    
    private void setDefenderParams(int count) {
    	if (count>BURNIN) {
    		for (int i=0;i<nrDefenders;i++) {
    			defenders.get(i).BURNIN = 1;
    		}
    	}
    }
    
    private void addDefenders(Double[][] qla, int[] rulescount, int count, Coordinates coord, String mf) {
    	for (int i=0;i<nrDefenders;i++){
            defenders.add(new Defender_FQL("FQL_d"+i, ALL_DOTS, B_WIDTH, B_HEIGHT, CAPTUREDIST, 
            		coord.Dx[i], coord.Dy[i], Cx, Cy, Tx, Ty, coord.Ix, coord.Iy, qla, rulescount, mf));    		
    	}
    	setDefenderParams(count);
    }
    
    public void addDefenders(double[] actor, double[] critic, int[] rulescount, int count, Coordinates coord, String mf) {
    	for (int i=0;i<nrDefenders;i++){
            defenders.add(new Defender_FACL("FACL_d"+i, ALL_DOTS, B_WIDTH, B_HEIGHT, CAPTUREDIST, 
            		coord.Dx[i], coord.Dy[i], Cx, Cy, Tx, Ty, coord.Ix, coord.Iy, actor, critic, rulescount, mf));    		
    	}
    	setDefenderParams(count);
    }
    
    public void addDefenders(Double[][] qla, Double[][] qlzeta, List<Double[][]> inputParams, 
    		List<Double[][]> eligibilityTrace, int[] rulescount, int count, Coordinates coord) {
    	for (int i=0;i<nrDefenders;i++){
            defenders.add(new Defender_QLFIS("QLFIS_d"+i, ALL_DOTS, B_WIDTH, B_HEIGHT, CAPTUREDIST, 
            		coord.Dx[i], coord.Dy[i], Cx, Cy, Tx, Ty, coord.Ix, coord.Iy, qla, qlzeta,
            		inputParams, eligibilityTrace, rulescount));
    	}
    	setDefenderParams(count);
    }
    
    private void locateTerritory() {
    	for (int theta=0;theta<Tx.length;theta++) {
    		Tx[theta] = Math.round((float) (Cx + R * Math.cos(2 * Math.PI / 360 * theta)));
    		Ty[theta] = Math.round((float)  (Cy + R * Math.sin(2 * Math.PI / 360 * theta)));
    	}
    }

    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);

        doDrawing(g);
    }
    
    private void doDrawing(Graphics g) {
    	
    	// draw territory center
    	g.drawImage(territoryCenter, Cx, Cy, this);
    	// draw territory boundary
    	for (int t=0; t<Tx.length; t++) {
    		if (Tx[t]==0 && Ty[t]==0) break;
    		g.drawImage(territory, Tx[t], Ty[t], this);
    	}
    	g.drawImage(nearestBorder, defenders.get(0).t[0], defenders.get(0).t[1], this);
    	
    	for (int i=0;i<nrDefenders;i++) {
        	for (int r=0; r<defenders.get(i).DRx.length; r++) {
        		if (defenders.get(i).DRx[r]==0 && defenders.get(i).DRy[r]==0) break;
        		g.drawImage(defenderRange, defenders.get(i).DRx[r], defenders.get(i).DRy[r], this);   		
        	}   		
    	}
    	
    	// draw players and their moving tracks
        for (int z=dots-1; z>=0; z--) {
            if (z == 0) {
            	for (int i=0;i<nrInvaders;i++)
            		g.drawImage(invaderNow, invaders.get(i).Ix[z]-DOT_SIZE/2, invaders.get(i).Iy[z]-DOT_SIZE/2, this);
            	for (int i=0;i<nrDefenders;i++)
            		g.drawImage(defenderNow, defenders.get(i).Dx[z]-DOT_SIZE/2, defenders.get(i).Dy[z]-DOT_SIZE/2, this);
            } else {
            	for (int i=0;i<nrInvaders;i++) {
            		if (z <= invaders.get(i).Ix.length-1)
            			g.drawImage(invaderTrack, invaders.get(i).Ix[z]-DOT_SIZE/2, invaders.get(i).Iy[z]-DOT_SIZE/2, this);
            	}
            	for (int i=0;i<nrDefenders;i++) {
            		if (z <= defenders.get(i).Dx.length-1)
            			g.drawImage(defenderTrack, defenders.get(i).Dx[z]-DOT_SIZE/2, defenders.get(i).Dy[z]-DOT_SIZE/2, this);
            	}	
            }
        }
        
        for (int i=0;i<nrInvaders;i++)
        	g.drawImage(intersect, invaders.get(i).Ox, invaders.get(i).Oy, this);
    }
    
    private void move() {
    	
    	for (int i=0;i<nrInvaders;i++)
    		invaders.get(i).move(dots);
    	int[] Ix0 = invaders.stream().filter(x->x.inGame).map(x->x.Ix[0]).mapToInt(Integer::intValue).toArray();
    	int[] Iy0 = invaders.stream().filter(x->x.inGame).map(x->x.Iy[0]).mapToInt(Integer::intValue).toArray();
    	if (Ix0.length > 0){
	    	for (int i=0;i<nrDefenders;i++)
	    			defenders.get(i).move(dots, Ix0, Iy0);
    	}
    	
//    	defender1.move(dots, invader.Ox, invader.Oy);
//    	defender2.move(dots, invader.Ox, invader.Oy);
    	
    	dots ++;
    }
    
    private boolean isCaptured(Invader i, Defender d) {
    	return Math.pow(i.Ix[0] - d.Dx[0], 2) + Math.pow(i.Iy[0] - d.Dy[0], 2) <= Math.pow(CAPTUREDIST, 2);
    }
    
    private void checkIntercept() {
    	
    	for (int i=0;i<nrInvaders;i++) {
    		for (int d=0;d<nrDefenders;d++) {
    			if ((invaders.get(i).inGame) && isCaptured(invaders.get(i),defenders.get(d))) {
    				defenders.get(d).winner = true;
    				invaders.get(i).inGame = false;
    				nrInvadersInGame --;
    			}
    		}
    		if ( (invaders.get(i).inGame) && (dots>1) && (invaders.get(i).distToTerritory <= R)) {
    			invaders.get(i).inGame = false;
    			nrInvadersInTerritory ++;
    			nrInvadersInGame --;
    		}   	
    	}
    	
		if (nrInvadersInGame<=0) {
			defenderWins = nrInvaders - nrInvadersInTerritory;
			setEndGame(true);
		}
    	
        if (getEndGame()) {
            timer.stop();
        }
    }

    @Override
    public void actionPerformed(ActionEvent e) {

        if (inGame && !getEndGame()) {
            checkIntercept();
            move();
        }
        
        repaint();
    }

    private class TAdapter extends KeyAdapter {

        @Override
        public void keyPressed(KeyEvent e) {

            int key = e.getKeyCode();

            if ((key == KeyEvent.VK_ENTER)) {
                inGame = !inGame;
            }
            
            if ((key == KeyEvent.VK_ESCAPE)) {
            	setEndGame(true);
            }
        }
    }
    
    private class Coordinates {
    	int[] Dx;
    	int[] Dy;
    	int[] Ix;
    	int[] Iy;
    	
    	public Coordinates(int[] Dx, int[] Dy, int[] Ix, int[] Iy) {
//    		DX = Arrays.stream(Dx).boxed().toArray(Integer[]::new);
    		this.Dx = Dx.clone();
    		this.Dy = Dy.clone();
    		this.Ix = Ix.clone();
    		this.Iy = Iy.clone();
    	}
    }
}

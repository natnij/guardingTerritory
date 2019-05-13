package territoryGame;

import java.util.Arrays;

public class Invader {
	
	String name;
    int[] Ix;
    int[] Iy;
    private int[] IxBest;
    private int[] IyBest;
    
    int D1x0;
    int D1y0;
    int D2x0;
    int D2y0;
    int Cx;
    int Cy;
    int[] Tx;
    int[] Ty;
    private int[] ID1O1;
    private int[] ID2O2;
    
    int Ox;
    int Oy;
    double distToTerritory;
    private int steps;
    boolean inGame = true;
    
    private int B_WIDTH;
    private int B_HEIGHT;
	
	public Invader(String name, int ALL_DOTS, int b_width, int b_height, int Ix0, int Iy0, 
				   int cx, int cy, int[] tx, int[] ty, int[] dx, int[] dy){
	    B_WIDTH = b_width;
	    B_HEIGHT = b_height;
		Ix = new int[ALL_DOTS];
	    Iy = new int[ALL_DOTS];
        Ix[0] = Ix0;
        Iy[0] = Iy0;
	    IxBest = new int[ALL_DOTS];
	    IyBest = new int[ALL_DOTS];
	    Cx = cx;
	    Cy = cy;
	    Tx = tx.clone();
	    Ty = ty.clone();
	    ID1O1 = new int[B_WIDTH];
	    ID2O2 = new int[B_WIDTH];
	    distToTerritory = calculateDist(Ix[0],Iy[0],Cx,Cy);
	    
	    selectDefenders(dx,dy);
	    initInvaderTrack();
	}
	
	private void selectDefenders(int[] dx, int[] dy) {
		
		ValContainer<Integer> idx = new ValContainer<Integer>(0);
		ValContainer<Double> bestDist = new ValContainer<Double>(calculateDist(Ix[0],Iy[0],Tx[0],Ty[0]));
		for (int i=1;i<Tx.length;i++) {
			double dist = calculateDist(Ix[0],Iy[0],Tx[i],Ty[i]);
			if (dist < bestDist.getVal()){
				bestDist.setVal(dist);
				idx.setVal(i);
			}
		}
		
		int tx = Tx[idx.getVal()];
		int ty = Ty[idx.getVal()];
		
		int[] selected;
		double[] d_dist;
		double d0_dist = calculateDist(tx,ty,dx[0],dy[0]);
		double d1_dist = calculateDist(tx,ty,dx[1],dy[1]);
		if (d0_dist <= d1_dist) {
			selected = new int[]{0,1};
			d_dist = new double[]{d0_dist,d1_dist};
		} else {
			selected = new int[]{1,0};
			d_dist = new double[]{d1_dist,d0_dist};			
		}

		for (int i=2; i<dx.length; i++) {
			double dist = calculateDist(Ix[0],Iy[0],dx[i],dy[i]);
			if (dist < d_dist[0]){
				selected[0] = i;
				d_dist[0] = dist;
			} else if (dist < d_dist[1]) {
				selected[1] = i;
				d_dist[1] = dist;				
			}
		}
		
	    D1x0 = dx[selected[0]];
	    D1y0 = dy[selected[0]];
	    D2x0 = dx[selected[1]];
	    D2y0 = dy[selected[1]];
	}
	
    private int[] getPerpendicular(int refx, int refy) {
    	double intersectx = (Ix[0] + refx) / 2;
    	double intersecty = (Iy[0] + refy) / 2;
    	int perpend_y[] = new int[B_WIDTH];
    	Arrays.fill(perpend_y, (int) intersecty);
    	double angle = Math.atan((Iy[0]-refy)*1.0/(Ix[0]-refx));
    	if (angle==0.0) {
    		Arrays.fill(perpend_y, -1);
    		return perpend_y;
    	}
    	
    	double alpha = angle + Math.PI/2;
    	for (int x=0;x<B_WIDTH;x++) {
    		perpend_y[x] = (int)( (x-intersectx) * Math.tan(alpha) + intersecty );
    	}
    	return perpend_y;
    }
    
    private void initInvaderTrack() {
    	IxBest[0] = Ix[0];
    	IyBest[0] = Iy[0];
    	
    	int[] id1o1 = getPerpendicular(D1x0,D1y0);
    	int[] id2o2 = getPerpendicular(D2x0,D2y0);
    	for (int i=0;i<ID1O1.length;i++) {
    		ID1O1[i] = id1o1[i];
    		ID2O2[i] = id2o2[i];
    	}
    	
    	if ((ID1O1[0]==-1) && (ID2O2[0]!=-1) ) {
    		Ox = Math.round((float) ((Ix[0] + D1x0) / 2));
    		Oy = ID2O2[Ox];
    	} else if ( (ID2O2[0]==-1) && (ID1O1[0]!=-1) ) {
    		Ox = Math.round((float) ((Ix[0] + D2x0) / 2));
    		Oy = ID1O1[Ox];
    	} else if ( (ID2O2[0]!=-1) && (ID1O1[0]!=-1) ) {
    		for (int x=0; x<ID1O1.length; x++) {
    			if(ID1O1[x]==ID2O2[x]) {
    				Ox = x;
    				Oy = ID1O1[x];
    				break;
    			}
    		}
    	} else {
    		Oy = Cy;
    		Ox = Cx;
    		int x1 = (int) ((Ix[0] + D1x0) / 2);
    		int x2 = (int) ((Ix[0] + D2x0) / 2);
    		if ( Math.abs(x1 - Ix[0]) < Math.abs(Ox - Ix[0]) ) Ox = x1;
    		if ( Math.abs(x2 - Ix[0]) < Math.abs(Ox - Ix[0]) ) Ox = x2;
    	}
    	
    	if ((Ox==0) && (Oy==0)) {
    		// all three players on the same line:
    		Ox = Cx;
    		Oy = Cy;
    		double dist2Init = Math.pow(Ox-Ix[0],2) + Math.pow(Oy-Iy[0],2);
    		
    		ValContainer<Integer> interc1_x = new ValContainer<Integer>();
    		ValContainer<Integer> interc1_y = new ValContainer<Integer>();
    		ValContainer<Double> dist2Interc1 = new ValContainer<Double>(Math.pow(B_WIDTH,2) + Math.pow(B_HEIGHT,2));
    		ValContainer<Integer> interc2_x = new ValContainer<Integer>();
    		ValContainer<Integer> interc2_y = new ValContainer<Integer>();
    		ValContainer<Double> dist2Interc2 = new ValContainer<Double>(Math.pow(B_WIDTH,2) + Math.pow(B_HEIGHT,2));
    		for (int x=0; x<ID1O1.length; x++) {
    			if ( (Math.pow(x-Cx,2) + Math.pow(ID1O1[x]-Cy,2)) <= dist2Interc1.getVal() ) {
    				interc1_x.setVal(x);
    				interc1_y.setVal(ID1O1[x]);
    				dist2Interc1.setVal(Math.pow(x-Cx,2) + Math.pow(ID1O1[x]-Cy,2));
    			}
    			if ( (Math.pow(x-Cx,2) + Math.pow(ID2O2[x]-Cy,2)) <= dist2Interc2.getVal() ) {
    				interc2_x.setVal(x);
    				interc2_y.setVal(ID2O2[x]);
    				dist2Interc2.setVal( Math.pow(x-Cx,2) + Math.pow(ID2O2[x]-Cy,2));
    			}
    		}
    		if ( (Math.pow(interc1_x.getVal()-Ix[0],2) + Math.pow(interc1_y.getVal()-Iy[0],2)) <= dist2Init ) {
    			Ox = interc1_x.getVal();
    			Oy = interc1_y.getVal();
    			dist2Init = Math.pow(interc1_x.getVal()-Ix[0],2) + Math.pow(interc1_y.getVal()-Iy[0],2);
    		}
    		if ( (Math.pow(interc2_x.getVal()-Ix[0],2) + Math.pow(interc2_y.getVal()-Iy[0],2)) <= dist2Init ) {
    			Ox = interc2_x.getVal();
    			Oy = interc2_y.getVal();
    		}
    		int[] newxy = adjustBoundaries(Ox, Oy);
    		Ox = newxy[0];
    		Oy = newxy[1];
    	}
    	
    	ValContainer<Integer> distInit2O = new ValContainer<Integer>( (int)Math.sqrt( Math.pow(Ox-Ix[0], 2) + Math.pow(Oy-Iy[0],2)) );
    	for (int i=1; i<Math.min(distInit2O.getVal(),IxBest.length); i++) {
    		IxBest[i] = Math.round((float) (i * 1.0 / distInit2O.getVal() * (Ox - Ix[0]) + Ix[0]));
    		IyBest[i] = Math.round((float) (i * 1.0 / distInit2O.getVal() * (Oy - Iy[0]) + Iy[0]));
    		int[] newxy = adjustBoundaries(IxBest[i], IyBest[i]);
    		IxBest[i] = newxy[0];
    		IyBest[i] = newxy[1];
    	}
    	steps = distInit2O.getVal();
    	ValContainer<Integer> distO2T = new ValContainer<Integer>( (int)Math.sqrt(Math.pow(Ox-Cx,2) + Math.pow(Oy-Cy,2)) );
    	for (int i=0; i<Math.min(distO2T.getVal(),IxBest.length-steps); i++) {
    		IxBest[steps-1+i] = Math.round((float)(i * 1.0 / distO2T.getVal() * (Cx-Ox) + Ox));
    		IyBest[steps-1+i] = Math.round((float) (i * 1.0 / distO2T.getVal() * (Cy-Oy) + Oy));
    		int[] newxy = adjustBoundaries(IxBest[steps-1+i], IyBest[steps-1+i]);
    		IxBest[steps-1+i] = newxy[0];
    		IyBest[steps-1+i] = newxy[1];
    	}
    	steps = steps + distO2T.getVal();
    	
    }
    
    private int[] adjustBoundaries(int x, int y) {
		int newx = Math.min(x, B_WIDTH-1);
		int newy = Math.min(y, B_HEIGHT-1);
		return new int[]{newx, newy};
    }
    
    public void move(int dots) {
    	
    	if (inGame) {
    		if (dots >= Ix.length) {
    			distToTerritory = -1;
    			inGame = false;
    			return;
    		}
    		
	        for (int z = dots; z > 0; z--) {
	            Ix[z] = Ix[(z - 1)];
	            Iy[z] = Iy[(z - 1)];
	        }
	        
	        calInvaderNEPos(dots);
	        distToTerritory = calculateDist(Ix[0],Iy[0],Cx,Cy);
    	}
    }
    
    private void calInvaderNEPos(int dots) {
    	if (dots<=steps) {
        	Ix[0] = IxBest[dots-1];
        	Iy[0] = IyBest[dots-1];
    	}
    }
    
    private double calculateDist(int originX, int originY, int destX, int destY) {
    	return Math.sqrt(Math.pow(originX - destX, 2) + Math.pow(originY - destY, 2));
    }
}

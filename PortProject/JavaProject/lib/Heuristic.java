package Environment;

import java.util.Random;

import simplifiedSimulator.State;

public class Heuristic {
	
	
	public Heuristic() {
		
	}
	public void showFeatures(State state) {
		int shipAmount = state.shipAmount;
		System.out.println("----------------------------------------");
		System.out.println("\nCurrent Working Amount");
		int currentWorkingSum = 0;
		for (int i=0;i<shipAmount*4;i++) {
			System.out.print(state.currentWorkingTruckAmount[i]+" ");
			currentWorkingSum += state.currentWorkingTruckAmount[i];
		}
		System.out.print(" Sum: " + currentWorkingSum);
		System.out.println("\nQueue Length");
		int queueLengthSum = 0;
		for (int i=0;i<shipAmount*4;i++) {
			System.out.print(state.qcTruckQueueLength[i]+" ");
			queueLengthSum += state.qcTruckQueueLength[i];
		}
		System.out.print(" Sum: " + queueLengthSum);
		System.out.println("\nHeading Amount: ");
		int headingSum =0;
		for (int i=0;i<shipAmount*4;i++) {
			int heading = state.headingToQCAmount[i];
			headingSum += state.headingToQCAmount[i];
			System.out.print(heading+" ");
		}
		System.out.print("Sum: " + headingSum);
		System.out.println("\nTask Remain");
		for (int i=0;i<shipAmount*4;i++) {
			System.out.print(state.qcRemainTaskAmount[i]+" ");
		}
		System.out.println("\nDistance to QC");
		for (int i=0;i<shipAmount*4;i++) {
			System.out.print(state.truckToQCDistance[i]+" ");
		}
		
		System.out.println("\nQC Type");
		for(int i=0;i<state.shipAmount*4;i++) {
			System.out.print(state.qcTypes[i]+" ");
		}
		System.out.println();
	}
	
	
	public int chenHeuristic(State state) {
		
		
		// =========== Heuristic ====================
		
		this.showFeatures(state);
		int min_amount = 1000;
		int min_index = 0;
		for (int i=0;i<state.shipAmount*4;i++) {
			if ((state.headingToQCAmount[i]<=min_amount) && (state.qcRemainTaskAmount[i] != 0)) {
				min_amount = state.headingToQCAmount[i];
				min_index = i;
			}
		}
		int action = min_index;
		
		//System.out.println(" action: "+action);
		//action = new Random().nextInt(info.shipAmount * 4);
		
		// random action
		//int action = new Random().nextInt(state.shipAmount * 4);
		
		return action;
	}
}






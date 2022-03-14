package TaskGenerator;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;

public class TaskGenerator {
	
	//private int taskAmount; 
	//private boolean taskAllocateEventoShips;
	
	private int shipAmount;	// max 7
	private int[] shipLoadContainerAmounts;		// max 2000
	private int[] shipDispatchContainerAmounts;  // max 1400
	//  71 load yards
	private int[] shipLoadYardBlockAmounts;			//  71 load yards
	private int[] shipDispatchYardBlockAmounts;		//  13 unload yards
	private double relocatedContainerAmount;
	public ArrayList<Task> taskList = new ArrayList<Task>();
	ArrayList<Yard> yards;
	
	private void feasibilityCheck() {
		if ((shipAmount<0) || (shipAmount>7)) {
			System.out.println("Infeasible Ship Amount");
			System.exit(-1);
		}
		if ((shipLoadContainerAmounts.length != shipAmount) || (shipDispatchContainerAmounts.length != shipAmount) ||
			 shipLoadYardBlockAmounts.length != shipAmount || (shipDispatchYardBlockAmounts.length != shipAmount)) {
			System.out.println("Array Length Unmatch Ship Amount!");
			System.exit(-1);
		}
		for (int i=0; i<shipAmount; i++) {
			
			if ((shipLoadContainerAmounts[i] < 0) || (shipLoadContainerAmounts[i] > 2000)) {
				System.out.println("Infeasible Load Container Amount for Ship "+i);
				System.exit(-1);
			}
			if ((shipDispatchContainerAmounts[i] < 0) || (shipDispatchContainerAmounts[i] > 1400)) {
				System.out.println("Infeasible Dispatch Container Amount for Ship "+i);
				System.exit(-1);
			}
			if (shipLoadContainerAmounts[i] > 750 * shipLoadYardBlockAmounts[i]) {
				System.out.println("Insufficient Load Yard Block for Ship "+i);
				System.exit(-1);
			}
			if (shipDispatchContainerAmounts[i] > 750 * shipDispatchYardBlockAmounts[i]) {
				System.out.println("Insufficient Dispatch Yard Block for Ship "+i);
				System.exit(-1);
			}
		}
	}
	
	public Yard chooseRandomYard(HashSet<Yard> selectedYards) {
		while(true) {
			int index = new Random().nextInt(selectedYards.size());
			
			// transfer to array
			Object[] yards = selectedYards.toArray();
			
			Yard y = (Yard)yards[index];
			if (y.reserved < 750) {
				y.reserved++;
				return y;
			}
		}
	}
	
	public HashSet<Yard> chooseSetOfYards(boolean isExport, int amount){
		//System.out.println(amount);
		HashSet<Yard> yardSet = new HashSet<Yard>();
		while(yardSet.size() < amount) {
			int index = new Random().nextInt(yards.size());
			Yard y = yards.get(index);
			if ((y.reserved < 750) && (y.isExport == isExport)) {
				yardSet.add(y);
			}
		}
		return yardSet;
	}
	
	private int yardBayAmount = 25;
	private int yardStackAmount = 6;
	private int shipBayAmount = 11;
	private int shipStackAmount = 20;
	public TwoDPoint chooseRandomPile(int type, int posIndex) {
		int bay;
		int stack;
		while(true) {
			 if (type == 0) {
				 bay = new Random().nextInt(shipBayAmount)+shipBayAmount;
				 stack = new Random().nextInt(shipStackAmount);
				 if (pileHeightCheck(type, posIndex, bay, stack)) {
					 return new TwoDPoint(bay, stack);
				 }
				 
			 } else {
				 bay = new Random().nextInt(yardBayAmount);
				 stack = new Random().nextInt(yardStackAmount);
				 if (pileHeightCheck(type, posIndex, bay, stack)) {
					 return new TwoDPoint(bay, stack);
				 }
			 }
		}
	}
	
	public TaskGenerator(int shipAmount, int[] shipLoadContainerAmounts, int[] shipDispatchContainerAmounts,
						 int[] shipLoadYardBlockAmounts, int[] shipDispatchYardBlockAmounts, int relocatedContainerAmount) {
		this.shipAmount = shipAmount;
		this.shipLoadContainerAmounts = shipLoadContainerAmounts;
		this.shipDispatchContainerAmounts = shipDispatchContainerAmounts;
		this.shipLoadYardBlockAmounts = shipLoadYardBlockAmounts;
		this.shipDispatchYardBlockAmounts = shipDispatchYardBlockAmounts;
		this.relocatedContainerAmount = relocatedContainerAmount;
		
		feasibilityCheck();
	}
	
	public boolean pileHeightCheck(int type, int posIndex, int bay, int stack) {
		int height = 0;
		for (Task t : taskList) {
			if ((type == t.type) && (posIndex == t.source) && (bay == t.bay) && (stack == t.stack)) {
				height++;
			}
		}
		return height < 6;
	}
	
	public void generate() {
		yards = new createYards().create();
		
		int id = 0;
		for (int ship=0; ship<shipAmount; ship++) {
			
			// load tasks
			int loadAmount = shipLoadContainerAmounts[ship];
			int loadYardAmount = shipLoadYardBlockAmounts[ship];
			
			//System.out.println(loadYardAmount);
			HashSet<Yard> seletedLoadYards = chooseSetOfYards(true, loadYardAmount);
			//System.out.println(seletedLoadYards.size());

			for(int j=0; j<loadAmount; j++) {
				int type = 1;
				Yard y = chooseRandomYard(seletedLoadYards);
				TwoDPoint pile = chooseRandomPile(type, y.index);
				Task t = new Task(id, y.index, ship, type, pile.x, pile.y, 2);
				taskList.add(t);
				id++;
			}
			
			// dispatch tasks
			int dispatchAmount = shipDispatchContainerAmounts[ship];
			int dispatchYardAmount = shipDispatchYardBlockAmounts[ship];
			HashSet<Yard> seletedDispatchYards = chooseSetOfYards(false, dispatchYardAmount);
			
			for (int k=0; k<dispatchAmount; k++) {
				int type = 0;
				Yard y = chooseRandomYard(seletedDispatchYards);
				TwoDPoint pile = chooseRandomPile(type, y.index);
				Task t = new Task(id, ship, y.index, type, pile.x, pile.y, 2);
				taskList.add(t);
				id++;
			}
			Collections.shuffle(taskList);
		}
	}
	 
	public void writeToFile(int shipAmount, String fileName) {
		ArrayList<Task> taskList = this.taskList;
		String path = "..\\Data\\";
		String name = fileName;
		
		try{
			FileWriter writer=new FileWriter(path+name+".txt");
			writer.write(shipAmount+"\n");
			for (Task t : this.taskList) {
				writer.write(t.id+","+t.source+","+t.destination+","+t.type+","+t.bay+","+t.stack+","+t.teu+"\n");
			}
			writer.close();
		}catch (Exception e){
			e.printStackTrace();
		}
	}
	
	
	public static void main(String[] args) {
		//task
		//id, source, destination, type(load/dispatch), bay, stack, TEU(2)
		int shipAmount = 4;
		int[] shipLoadContainerAmounts = new int[]{50,50,50,50};
		int[] shipDispatchContainerAmounts = new int[]{50,50,50,50};
		int[] shipLoadYardBlockAmounts = new int[] {3,3,3,3};
		int[] shipDispatchYardBlockAmounts = new int[] {3,3,3,3};
		int relocatedContainerAmount = 0;
		
		TaskGenerator tg = new TaskGenerator(shipAmount,
											 shipLoadContainerAmounts,
											 shipDispatchContainerAmounts,
											 shipLoadYardBlockAmounts,
											 shipDispatchYardBlockAmounts,
											 relocatedContainerAmount);
		tg.generate();
		
		// show task info
		for (Task t : tg.taskList) {
			System.out.println(t.toString());
		}
		tg.writeToFile(shipAmount, "4Ship_small");
	}
}







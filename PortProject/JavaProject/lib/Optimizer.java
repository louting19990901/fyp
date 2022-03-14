package Environment;

public class Optimizer {
	
	public void localSearch(int[] solution) {
		
	}
	
	public int[] generateInitSolution() {
		Env env = Env.getInstance();
		Env.collectMode = true;
		Env.optimizeMode = false;
		int episodeLength = 400;
		Env.actionSequence = new int[episodeLength];
		env.runFastMode();
		int[] solution = Env.actionSequence.clone();
		return solution;
	}
	
	public void collectEpisode() {
		Env.collectMode = false;
		Env.optimizeMode = true;
		
		
	}

	public static void main(String[] argss) {
		Optimizer opt = new Optimizer();
		int[] solution = opt.generateInitSolution();
		for(int i=0; i<solution.length;i++) {
			System.out.print(solution[i]+" ");
		}
		System.out.println("\nLength: "+solution.length);
	}
	 	
	
}

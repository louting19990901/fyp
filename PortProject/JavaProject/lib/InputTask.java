package Environment;

public class InputTask {
	
	public String source;	// berth or yard
	public String destinaton; //berth or yard
	public String taskType; // DSCH or LOAD
	public String targetQC;
	
	public InputTask(String source, String destination, String taskType, String targetQC) {
		this.source = source;
		this.destinaton = destination;
		this.taskType = taskType;
		this.targetQC = targetQC;
	}
	
	
}

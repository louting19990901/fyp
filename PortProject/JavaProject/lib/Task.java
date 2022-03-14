package TaskGenerator;

public class Task {
	
	// id, source, destination, type(load/dispatch), source_bay, source_stack, 
	
	public int id;
	public int source;
	public int destination;
	public int type; // 0:dispatch 1:load
	public int bay;
	public int stack;
	public int teu;
	
	public Task(int id, int source, int destination, int type, int bay, int stack, int teu) {
		this.id = id;
		this.source = source;
		this.destination =destination;
		this.type = type;
		this.bay = bay;
		this.stack = stack;
		this.teu = teu;
	}

	@Override
	public String toString() {
		return "Task [id=" + id + ", source=" + source + ", destination=" + destination + ", type=" + type + ", bay="
				+ bay + ", stack=" + stack + ", teu=" + teu + "]";
	}

	
	
}

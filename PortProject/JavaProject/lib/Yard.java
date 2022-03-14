package TaskGenerator;

public class Yard {
	public int index;
	public boolean isExport = true;
	public int reserved = 0;
	public Yard(int index) {
		this.index = index;
	}
	
	public void addAttributes(boolean isExport) {
		this.isExport = isExport;
	}
}

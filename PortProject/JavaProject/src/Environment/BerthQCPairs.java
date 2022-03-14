package Environment;

public class BerthQCPairs {
	
	public String berth;
	public String qc;
	public BerthQCPairs(String berth, String qc) {
		this.berth = berth;
		this.qc = qc;
	}
	
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((berth == null) ? 0 : berth.hashCode());
		result = prime * result + ((qc == null) ? 0 : qc.hashCode());
		return result;
	}
	
	@Override
	public boolean equals(Object obj) {
		if (obj instanceof BerthQCPairs) {
			BerthQCPairs bqp = (BerthQCPairs)obj;
			return ((bqp.berth.equals(this.berth)) && (bqp.qc.equals(this.qc)));
		} else {
			return false;
		}
	}
}

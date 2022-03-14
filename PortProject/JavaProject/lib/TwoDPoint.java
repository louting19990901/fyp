package TaskGenerator;

public class TwoDPoint implements Comparable<Object> {

	public int x;
	public int y;
	public int pile0or1;
	
	public TwoDPoint(int x, int y) {
		this.x = x;
		this.y = y;
	}

	@Override
	public int compareTo(Object o) {
		TwoDPoint p = (TwoDPoint)o;
		if (this.x > p.x) {
			return 1;
		} else if (this.x < p.x) {
			return -1;
		} else {
			if (this.y > p.y) {
				return 1;
			} else if (this.y < p.y) {
				return -1;
			} else {
				return 0;
			}
		}
	}		
	
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + x;
		result = prime * result + y;
		return result;
	}


	@Override
	public boolean equals(Object obj) {
		if (obj instanceof TwoDPoint){
			TwoDPoint p = (TwoDPoint)obj;
			return (p.x==this.x) && (p.y==this.y);
		} else {
			return false;
		}
			
	}

	@Override
	public String toString() {
		return super.toString();
	}


}

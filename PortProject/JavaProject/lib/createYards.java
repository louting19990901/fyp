package TaskGenerator;

import java.util.ArrayList;

public class createYards {
	
	public ArrayList<Yard> yards = new ArrayList<Yard>();
	public ArrayList<Yard> create() {
		for (int i=0; i<97;i++) {
			yards.add(new Yard(i));
		}
		yards.get(15).addAttributes(false) ; 
		yards.get(17).addAttributes(false) ; 
		yards.get(18).addAttributes(false) ; 
		yards.get(19).addAttributes(false) ;  
		yards.get(23).addAttributes(false) ; 
		yards.get(50).addAttributes(false) ; 
		yards.get(53).addAttributes(false) ; 
		yards.get(55).addAttributes(false) ; 
		yards.get(65).addAttributes(false) ; 
		yards.get(66).addAttributes(false) ; 
		yards.get(67).addAttributes(false) ; 
		yards.get(79).addAttributes(false) ; 
		yards.get(82).addAttributes(false) ; 
		
		return this.yards;
	}
}	
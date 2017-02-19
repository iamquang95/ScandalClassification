/**
 * (C) Le Hong Phuong, phuonglh@gmail.com
 *  Vietnam National University, Hanoi, Vietnam.
 */
package vn.hus.nlp.graph.util;

import vn.hus.nlp.graph.AdjacencyMatrixGraph;
import vn.hus.nlp.graph.Edge;
import vn.hus.nlp.graph.IGraph;

/**
 * @author Le Hong Phuong, phuonglh@gmail.com
 *         <p>
 *         Oct 26, 2007, 10:40:11 PM
 *         <p>
 *         Compute the transitive closure of a graph using Warhsall's algorithm.
 *         The transitive closure of a graph is usually represented by a dense
 *         graph, which is implemented by an adjacency matrix one.
 * 
 * @see AdjacencyMatrixGraph
 */
public class GraphTransitiveClosure {

	/**
	 * Get the transitive closure of a graph.
	 * 
	 * @param g a graph.
	 * @return the transitive closure of <code>g</code>. 
	 */
	public static IGraph getTransitiveClosure(IGraph g) {
		// copy the original graph
		IGraph transitiveClosure = GraphUtilities.copy(g, true);
		int n = g.getNumberOfVertices();
		// add dummy loop edges
		for (int u = 0; u < n; u++) {
			transitiveClosure.insert(new Edge(u, u));
		}
		// the Warhall's algorithm to compute the transitive closure
		for (int v = 0; v < n; v++)
			for (int u = 0; u < n; u++)
				if (transitiveClosure.edge(u, v)) {
					for (int w = 0; w < n; w++)
						if (transitiveClosure.edge(v, w))
							transitiveClosure.insert(new Edge(u, w));
				}
		return transitiveClosure;
	}

}

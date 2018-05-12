package sc.fiji.maskrcnn;

import java.io.IOException;

import org.scijava.io.location.FileLocation;
import org.scijava.io.location.Location;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.tensorflow.Graph;
import org.tensorflow.Session;

import net.imagej.tensorflow.TensorFlowService;

public abstract class AbstractPredictor {

	@Parameter
	protected LogService log;

	@Parameter
	protected TensorFlowService tfService;

	protected Graph graph;
	protected Session session;

	protected void loadModel(String modelURL, String modelName) {
		this.loadModel(modelURL, modelName, modelName);
	}

	protected void loadModel(String modelURL, String modelName, String modelFilename) {

		final Location modelLocation = new FileLocation(modelURL);

		try {
			this.graph = tfService.loadGraph(modelLocation, modelName, modelFilename);
			this.session = new Session(this.graph);
		} catch (IOException e) {
			log.error(e);
		}
	}

	protected void clear() {
		// Do some cleaning
		this.session.close();
		tfService.dispose();
	}

	public Graph getGraph() {
		return graph;
	}

	public Session getSession() {
		return session;
	}

}

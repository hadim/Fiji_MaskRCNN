
package sc.fiji.maskrcnn.tracking;

import net.imagej.Dataset;
import net.imagej.table.GenericTable;

import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

@Plugin(type = Command.class, headless = true)
public class ObjectTracker implements Command {

	@Parameter
	private LogService log;

	@Parameter
	private GenericTable table;

	@Parameter
	private Dataset mask;

	@Override
	public void run() {

	}

}

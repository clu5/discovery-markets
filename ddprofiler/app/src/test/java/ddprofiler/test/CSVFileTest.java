package ddprofiler.test;

import ddprofiler.sources.implementations.CSVSource;
import ddprofiler.sources.config.CSVSourceConfig;
import ddprofiler.preanalysis.PreAnalyzer;
import ddprofiler.sources.deprecated.Attribute;
import org.junit.Test;

import java.io.IOException;
import java.sql.SQLException;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertNotNull;

public class CSVFileTest {

    @Test
    public void testCSVFile() throws IOException, SQLException {
        // Configure the CSV source
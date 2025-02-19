package ddprofiler.sources.implementations;

import static com.codahale.metrics.MetricRegistry.name;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.RFC4180Parser;
import com.opencsv.RFC4180ParserBuilder;
import com.opencsv.exceptions.CsvValidationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.codahale.metrics.Counter;

import metrics.Metrics;
import ddprofiler.sources.Source;
import ddprofiler.sources.SourceType;
import ddprofiler.sources.SourceUtils;
import ddprofiler.sources.config.CSVSourceConfig;
import ddprofiler.sources.config.SourceConfig;
import ddprofiler.sources.deprecated.Attribute;
import ddprofiler.sources.deprecated.Record;

public class CSVSource implements Source {

    final private Logger LOG = LoggerFactory.getLogger(CSVSource.class.getName());

    private int tid;
    private String path;
    private String relationName;
    private CSVSourceConfig config;
    private boolean initialized = false;
    private CSVReader fileReader;
    // private TableInfo tableInfo;
    private List<Attribute> attributes;

    // metrics
    private long lineCounter = 0;
    private Counter error_records = Metrics.REG.counter((name(CSVSource.class, "error", "records")));
    private Counter success_records = Metrics.REG.counter((name(CSVSource.class, "success", "records")));

    public CSVSource() {

    }

    public CSVSource(String path, String relationName, SourceConfig config) {
        this.tid = SourceUtils.computeTaskId(path, relationName);
        this.path = path;
        this.relationName = relationName;
        this.config = (CSVSourceConfig) config;
    }

    @Override
    public String getPath() {
        return path;
    }

    @Override
    public String getRelationName() {
        return relationName;
    }

    @Override
    public SourceConfig getSourceConfig() {
        return this.config;
    }

    @Override
    public int getTaskId() {
        return tid;
    }

    @Override
    public List<Source> processSource(SourceConfig config) {
        assert (config instanceof CSVSourceConfig);

        this.config = (CSVSourceConfig) config;

        List<Source> tasks = new ArrayList<>();

        CSVSourceConfig csvConfig = (CSVSourceConfig) config;
        String pathToSources = csvConfig.getPath();

        // TODO: at this point we'll be harnessing metadata from the source

        File folder = new File(pathToSources);
        int totalFiles = 0;
        int tt = 0;

        File[] filePaths = folder.listFiles();
        if (filePaths == null) {
            LOG.error("The path {} does not exist or is not a directory.", pathToSources);
            return tasks;
        }
        for (File f : filePaths) {
            tt++;
            if (f.isFile()) {
                String path = f.getParent() + File.separator;
                String name = f.getName();
                // Make the csv config specific to the relation
                CSVSource task = new CSVSource(path, name, config);
                totalFiles++;
                // c.submitTask(pt);
                tasks.add(task);
            }
        }

        LOG.info("Total files submitted for processing: {} - {}", totalFiles, tt);
        return tasks;
    }

    @Override
    public SourceType getSourceType() {
        return SourceType.csv;
    }

    @Override
    public List<Attribute> getAttributes() throws IOException, SQLException, CsvValidationException {
        if (!initialized) {
            String path = this.path + this.relationName;
            char separator = this.config.getSeparator().charAt(0);
            RFC4180Parser parser = new RFC4180ParserBuilder().withSeparator(separator).build();
            fileReader = new CSVReaderBuilder(new FileReader(path)).withCSVParser(parser).build();
            initialized = true;
        }
        // assume that the first row is the attributes;
        if (lineCounter != 0) {
            // return tableInfo.getTableAttributes();
            return attributes;
        }
        String[] attributeNames = fileReader.readNext();
        if (attributeNames == null) {
            throw new IOException("The CSV file " + this.path + this.relationName + " does not contain a header row.");
        }
        lineCounter++;

        List<Attribute> attrList = new ArrayList<Attribute>();
        for (int i = 0; i < attributeNames.length; i++) {
            Attribute attr = new Attribute(attributeNames[i]);
            attrList.add(attr);
        }
        this.attributes = attrList;
        return attrList;
    }

    @Override
    public Map<Attribute, List<String>> readRows(int num) throws IOException, SQLException, CsvValidationException {
        if (!initialized) {
            String path = this.path + this.relationName;
            char separator = this.config.getSeparator().charAt(0);
            RFC4180Parser parser = new RFC4180ParserBuilder().withSeparator(separator).build();
            fileReader = new CSVReaderBuilder(new FileReader(path)).withCSVParser(parser).build();
            initialized = true;
        }

        Map<Attribute, List<String>> data = new LinkedHashMap<>();
        // Make sure attrs is populated, if not, populate it here
        if (data.isEmpty()) {
            List<Attribute> attrs = this.getAttributes();
            attrs.forEach(a -> data.put(a, new ArrayList<>()));
        }

        // Read data and insert in order
        List<Record> recs = new ArrayList<>();
        boolean readData = this.read(num, recs);
        if (!readData) {
            return null;
        }

        for (Record r : recs) {
            List<String> values = r.getTuples();
            int currentIdx = 0;
            if (values.size() != data.values().size()) {
                error_records.inc();
                continue; // Some error while parsing data, a row has a
                // different format
            }
            success_records.inc();
            for (List<String> vals : data.values()) { // ordered iteration
                vals.add(values.get(currentIdx));
                currentIdx++;
            }
        }
        return data;
    }

    private boolean read(int numRecords, List<Record> rec_list) throws IOException, CsvValidationException {
        boolean read_lines = false;
        String[] res = null;
        for (int i = 0; i < numRecords && (res = fileReader.readNext()) != null; i++) {
            lineCounter++;
            read_lines = true;
            Record rec = new Record();
            rec.setTuples(res);
            rec_list.add(rec);
        }
        return read_lines;
    }

    @Override
    public void close() {
        try {
            fileReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}

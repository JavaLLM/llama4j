package org.javallm.llama4j.conf;

import java.util.Properties;

public interface ConfigParser {
    <T> T parse(Properties config);
}

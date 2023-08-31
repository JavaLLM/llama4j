package org.javallm.llama4j;

import java.io.File;

public class TestUtils {
    public static String getResourceAbsolutePath(String resource) {
        File f = new File(TestUtils.class.getClassLoader().getResource(resource).getFile());
        return f.getAbsolutePath();
    }
}

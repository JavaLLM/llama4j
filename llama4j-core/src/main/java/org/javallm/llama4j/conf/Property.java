package org.javallm.llama4j.conf;

import java.lang.annotation.*;

@Retention(RetentionPolicy.RUNTIME)
@Documented
@Target(ElementType.FIELD)
public @interface Property {
    String value();

    String key();
}

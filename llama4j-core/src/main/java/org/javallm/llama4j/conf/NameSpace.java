package org.javallm.llama4j.conf;

import java.lang.annotation.*;

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
@Documented
public @interface NameSpace {
    String value();
}

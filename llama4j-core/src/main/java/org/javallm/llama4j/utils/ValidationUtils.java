package org.javallm.llama4j.utils;


import jakarta.validation.ConstraintViolation;
import jakarta.validation.Validation;
import jakarta.validation.Validator;
import jakarta.validation.ValidatorFactory;
import org.apache.commons.collections4.CollectionUtils;
import org.hibernate.validator.HibernateValidator;
import org.hibernate.validator.messageinterpolation.ParameterMessageInterpolator;

import java.util.Set;

public final class ValidationUtils {
    private static final Validator validator;

    static {
        try (ValidatorFactory factory = Validation
                .byProvider(HibernateValidator.class)
                .configure()
                .messageInterpolator(new ParameterMessageInterpolator())
                .buildValidatorFactory()) {
            validator = factory.getValidator();
        }
    }

    private ValidationUtils() {
    }

    public static <T> void validateOrThrow(T bean, Class<?>... groups) {
        Set<ConstraintViolation<T>> violations = validator.validate(bean, groups);
        StringBuilder builder = new StringBuilder();
        if (CollectionUtils.isNotEmpty(violations)) {
            for (ConstraintViolation<T> violation : violations) {
                builder.append(violation.getMessage());
            }
            throw new IllegalArgumentException(builder.toString());
        }
    }

}

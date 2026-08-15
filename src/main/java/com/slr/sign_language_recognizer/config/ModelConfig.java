package com.slr.sign_language_recognizer.config;

import ai.onnxruntime.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import jakarta.annotation.PreDestroy;

@Configuration
public class ModelConfig {

    private OrtEnvironment env;
    private OrtSession session;

    @Value("${model.path}")
    private String modelPath;

    @Bean
    public OrtEnvironment ortEnvironment() {
        this.env = OrtEnvironment.getEnvironment();
        return this.env;
    }

    @Bean(destroyMethod = "close")
    public OrtSession ortSession(OrtEnvironment env) throws OrtException {
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        this.session = env.createSession(modelPath, opts);
        return this.session;
    }

    @PreDestroy
    public void cleanup() throws OrtException {
        if (session != null) session.close();
        if (env != null) env.close();
    }
}
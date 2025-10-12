#ifndef LOGGER_H
#define LOGGER_H

#include <NvInfer.h>
#include <iostream>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
};

extern Logger gLogger;

#endif // LOGGER_H

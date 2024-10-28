#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>


// 函数封装：初始化 CUPTI 并启用回调
// 函数封装：初始化 CUPTI 并启用回调
void initCUPTI(CUpti_SubscriberHandle *subscriber, CUptiResult *res, CUpti_CallbackFunc callbackFunc) {
    // 订阅 CUPTI 回调
    *res = cuptiSubscribe(subscriber, callbackFunc, NULL);  // 传递指针
    if (*res != CUPTI_SUCCESS) {
        const char *errstr;
        cuptiGetResultString(*res, &errstr);
        fprintf(stderr, "Failed to subscribe to CUPTI: %s\n", errstr);
        exit(EXIT_FAILURE);
    }

    // 启用 CUPTI 回调，拦截 Driver API 的 cuLaunchKernel
    *res = cuptiEnableCallback(1, *subscriber, CUPTI_CB_DOMAIN_DRIVER_API,  // 不需要解引用
                               CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
    if (*res != CUPTI_SUCCESS) {
        const char *errstr;
        cuptiGetResultString(*res, &errstr);
        fprintf(stderr, "Failed to enable CUPTI callback: %s\n", errstr);
        exit(EXIT_FAILURE);
    }

    printf("CUPTI initialized and callback enabled.\n");
}

// 函数定义：取消 CUPTI 订阅
void unsubscribeCUPTI(CUpti_SubscriberHandle *subscriber, CUptiResult *res) {
    *res = cuptiUnsubscribe(*subscriber);  // 使用指针操作 res 和 subscriber
    if (*res != CUPTI_SUCCESS) {
        const char *errstr;
        cuptiGetResultString(*res, &errstr);
        fprintf(stderr, "Failed to unsubscribe CUPTI: %s\n", errstr);
        exit(EXIT_FAILURE);
    } else {
        printf("CUPTI unsubscribed successfully.\n");
    }
}
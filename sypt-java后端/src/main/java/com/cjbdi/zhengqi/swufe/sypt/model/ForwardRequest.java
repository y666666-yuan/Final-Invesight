package com.cjbdi.zhengqi.swufe.sypt.model;

import com.cjbdi.zhengqi.swufe.sypt.prop.config.ForwardingConfig;
import lombok.Data;

import java.io.Serializable;
import java.net.URL;

@Data
public class ForwardRequest implements Serializable {

    private URL targetUrl;

    private ForwardingConfig.Route route;

}

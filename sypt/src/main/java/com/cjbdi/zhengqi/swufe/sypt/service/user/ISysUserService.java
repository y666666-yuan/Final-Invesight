package com.cjbdi.zhengqi.swufe.sypt.service.user;

import com.cjbdi.zhengqi.swufe.sypt.entity.user.SysUser;
import com.baomidou.mybatisplus.extension.service.IService;

/**
 * <p>
 * 系统用户表 服务类
 * </p>
 *
 * @author liangboning
 * @since 2025-05-27
 */
public interface ISysUserService extends IService<SysUser> {

    SysUser selectByUsername(String username);

}

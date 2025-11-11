package com.cjbdi.zhengqi.swufe.sypt.service.user.impl;

import cn.hutool.core.collection.CollUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.cjbdi.zhengqi.swufe.sypt.entity.user.SysUser;
import com.cjbdi.zhengqi.swufe.sypt.mapper.user.SysUserMapper;
import com.cjbdi.zhengqi.swufe.sypt.service.user.ISysUserService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * <p>
 * 系统用户表 服务实现类
 * </p>
 *
 * @author liangboning
 * @since 2025-05-27
 */
@Service
public class SysUserServiceImpl extends ServiceImpl<SysUserMapper, SysUser> implements ISysUserService {

    @Override
    public SysUser selectByUsername(String username) {
        LambdaQueryWrapper<SysUser> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(SysUser::getUsername, username);
        List<SysUser> list = list(queryWrapper);
        return CollUtil.isNotEmpty(list) ? list.get(0) : null;
    }

}

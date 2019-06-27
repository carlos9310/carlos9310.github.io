---
layout: page
title: About
description: my blog using jekyll 
repoistor:
keywords: comsince
comments: true
menu: 关于
permalink: /about/
---

## 联系

* GitHub：[https://github.com/carlos9310](https://github.com/carlos9310)
* 邮箱：carlos56059310@gmail.com


## 技能

#### 软件工程
<div class="btn-inline">
    {% for keyword in site.skill_software_keywords %}
    <button class="btn btn-outline" type="button">{{ keyword }}</button>
    {% endfor %}
</div>

#### 正在关注
<div class="btn-inline">
    {% for keyword in site.skill_update_keywords %}
    <button class="btn btn-outline" type="button">{{ keyword }}</button>
    {% endfor %}
</div>
 

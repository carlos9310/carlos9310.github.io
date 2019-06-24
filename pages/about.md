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
* 博客：[{{ site.title }}]({{ site.url }})


## Skill Keywords

#### Software Engineer Keywords
<div class="btn-inline">
    {% for keyword in site.skill_software_keywords %}
    <button class="btn btn-outline" type="button">{{ keyword }}</button>
    {% endfor %}
</div>

<!-- #### Mobile Developer Keywords
<div class="btn-inline">
    {% for keyword in site.skill_mobile_app_keywords %}
    <button class="btn btn-outline" type="button">{{ keyword }}</button>
    {% endfor %}
</div>

#### Windows Developer Keywords
<div class="btn-inline">
    {% for keyword in site.skill_windows_keywords %}
    <button class="btn btn-outline" type="button">{{ keyword }}</button>
    {% endfor %}
</div> -->

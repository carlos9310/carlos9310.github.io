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

## contact

* GitHub：[https://github.com/carlos9310](https://github.com/carlos9310)
* Email：carlos56059310@gmail.com


## skill

#### software engineering
<div class="btn-inline">
    {% for keyword in site.skill_software_keywords %}
    <button class="btn btn-outline" type="button">{{ keyword }}</button>
    {% endfor %}
</div>

#### following
<div class="btn-inline">
    {% for keyword in site.skill_update_keywords %}
    <button class="btn btn-outline" type="button">{{ keyword }}</button>
    {% endfor %}
</div>
 

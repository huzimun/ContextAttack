# 代码文件说明
## DeContext
- attack_decontext.py是DeContext初始版的代码
- attack_decontext_v2.py是DeContext本地跑通的代码
- attack_decontext_v2_commented.py是对attack_decontext_v2注释后的代码
## ContextAttack
- context_attack_v1.py是GPT-4o生成的代码，包括了三种语义对齐损失和DeContext中的上下文比例抑制损失。但是它的attention map损失对齐的是Q/K feature，是语义表征对齐
- context_attack_v2.py是ChatGPT生成的代码，包括了三种语义对齐损失和DeContext中的上下文比例抑制损失。但是它的attention map损失对齐的是attention map，是语义行为对齐
- context_attack_v3.py是ChatGPT改进后的完整版本，结合v1和v2，包括了4种语义对齐损失和1种上下文抑制损失
    - ChatGPT给了很多坑，当前需要让他把代码里 FLUX latent packing / target-context 拼接 也按原始 DeContext 的方式补回去。
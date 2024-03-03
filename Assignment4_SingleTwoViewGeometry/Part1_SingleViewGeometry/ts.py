original_list = ['0801-lm1-others']

# 将字符串拆分成单个字符
char_list = list(original_list[0])

# 用于映射替换数字1的字典
replace_dict = {'1': ['1', '2', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15']}

# 进行替换
new_list = [replace_dict.get(char, [char])[0] for char in char_list]

# 将结果拼接成字符串
result_str = ''.join(new_list)

# 将结果转换为list
result_list = [result_str]

print(result_list)
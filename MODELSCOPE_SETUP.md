# ModelScope API 配置说明

## 获取API Key

1. 访问 https://modelscope.cn/
2. 登录或注册账号
3. 进入个人中心
4. 找到"API Keys"或"访问令牌"
5. 创建新的API Key

## 配置步骤

1. 编辑 `config.env` 文件
2. 将 `your_modelscope_api_key_here` 替换为你的真实API Key

```env
MODELSCOPE_API_KEY=你的真实API_Key
MODELSCOPE_MODEL=Qwen/Qwen2.5-7B-Instruct
```

## 推荐模型

- `Qwen/Qwen2.5-7B-Instruct` - 通义千问2.5 7B（推荐）
- `Qwen/Qwen2.5-14B-Instruct` - 通义千问2.5 14B
- `Qwen/Qwen2.5-72B-Instruct` - 通义千问2.5 72B

## 使用方式

配置完成后，重启应用即可自动使用ModelScope API生成答案。

如果API调用失败，系统会自动回退到模板生成模式。

## 注意事项

- API Key请妥善保管，不要提交到代码仓库
- ModelScope API可能有调用次数限制
- 首次调用可能需要较长时间加载模型

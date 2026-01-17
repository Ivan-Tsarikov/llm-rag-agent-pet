# llm-rag-agent-pet

## MCP

Минимальная совместимая реализация MCP-инструментов (HTTP + JSON), чтобы показать
экспонирование инструментов, клиентское подключение и базовые меры безопасности.

### Запуск MCP server

```bash
python -m scripts.run_mcp_server
```

Переменные окружения:
- `MCP_HOST` (по умолчанию `0.0.0.0`)
- `MCP_PORT` (по умолчанию `9001`)
- `MCP_URL` (для клиента, по умолчанию `http://localhost:9001`)

### Проверка MCP-инструментов

```bash
python -m scripts.demo_mcp_tools
```

Ожидаемое поведение:
- `calc("3.5% * 12000")` возвращает значение около `420`.
- `search_docs("как восстановить доступ к аккаунту?")` показывает `account_security.*`
  среди top источников.
- `Проверка на лимиты (calc)` возвращает `{'error': 'Expression too long (max 200 chars).'}`
- `Проверка на лимиты (search_docs)` возвращает `{'error': 'Query too long (max 2000 chars).'}`

### Агент с MCP backend

```bash
$env:TOOL_BACKEND="mcp"
python -m scripts.demo_agent_mcp
```

В логах и выводе будет видно, что инструменты вызываются через MCP
(`tool_backend=mcp tool=...`). Этот шаг доказывает:

- MCP server экспонирует инструменты и лимиты.
- MCP client корректно их вызывает.
- Агент работает с переключателем backend’а (local/mcp).
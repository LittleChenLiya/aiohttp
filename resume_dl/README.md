# 断点下载组件（基于 aiohttp ClientSession）

一个小型单文件断点下载器：下载中途断开后，依据本地已确认的字节数与强 ETag
续传；只有完整文件落盘并校验通过后才发布（原子替换）。

## 恢复规则

1. **状态文件**。每次下载在目标文件旁写一个 `<目标名>.resume.json`，内容为
   JSON：`version`、`temp`（临时文件 basename）、`bytes`（已确认字节数）、
   `etag`（强凭据或 `null`）、`url`。状态与临时文件必须**一起校验**，任一不
   通过则视为无效状态，另建全新临时文件从头下载：
   - JSON 格式错误或版本不符；
   - `temp` 不是合法 basename、临时文件不存在；
   - 临时文件实际大小与 `bytes` 不一致；
   - `etag` 是弱 ETag 或格式非法；
   - `url` 与本次请求地址不一致。

2. **Range 与 If-Range**。续传时请求头携带
   `Range: bytes=<已确认字节数>-`；仅当持有**强 ETag** 时才携带
   `If-Range: <etag>`。弱 ETag（`W/"..."`）不能用作恢复凭据，不会出现在
   `If-Range` 中。

3. **206 Partial Content**。必须同时满足：
   - `Content-Range` 形如 `bytes <start>-<end>/<total>`，格式非法直接失败；
   - `start` 等于本地已确认字节数；
   - 实际收到字节数等于 `end - start + 1`；
   - 若给出 `total`，临时文件最终大小等于 `total`。
   段内容以追加方式写入既有临时文件。

4. **200 OK**。表示服务器忽略了 Range。此时**另建一个全新临时文件**从零写
   入，绝不接到旧片段后面；成功后删除旧临时文件与状态文件。

5. **中途断开**。读取过程中若连接被重置/关闭
   （`aiohttp.ClientPayloadError`、`aiohttp.ServerDisconnectedError` 等），把
   已收到的字节刷盘，原子写入恢复状态（已确认字节数 + ETag），抛出
   `ResumeInterrupted`。再次调用 `download()` 即可从断点续传。

6. **硬失败**。`Content-Range` 格式非法、范围起点不符、干净短读（服务器在
   未断开的情况下发送少于承诺的字节数）等情况抛出 `ResumeError`：不发布、不
   替换，**原来已发布的目标文件保持不动**；206 路径下会把临时文件截断回已确
   认长度以保持状态一致。

7. **发布**。完整文件写入并 `fsync` 后，通过 `os.replace` 原子替换目标文件，
   随后删除状态文件并 `fsync` 目录。

## 运行

```bash
# 演示（90 秒内完成，进程内 aiohttp 服务）
python resume_dl/demo.py

# 局部 pytest（禁用仓库重型 addopts）
pytest -o addopts="-p no:aiohttp" tests/test_resume_downloader.py
```

## 接口

```python
from aiohttp import ClientSession
from resume_dl import download, ResumeError, ResumeInterrupted

async with ClientSession() as session:
    try:
        await download(session, "https://example.com/file.bin", "/path/file.bin")
    except ResumeInterrupted as exc:
        # 进度已保留，可稍后重试
        print(exc.confirmed_bytes, exc.etag)
```

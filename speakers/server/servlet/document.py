from starlette.responses import RedirectResponse
from starlette.responses import HTMLResponse
from speakers.common.registry import registry


async def page_index():
    html_file = open(f"{registry.get_path('server_library_root')}/static/index.html", 'r').read()
    return HTMLResponse(html_file)


async def document():
    return RedirectResponse(url="/docs")

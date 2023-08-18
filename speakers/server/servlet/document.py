from starlette.responses import RedirectResponse


async def document():
    return RedirectResponse(url="/docs")

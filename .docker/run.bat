@echo off
REM ================================================================
REM run.bat – Windows-CMD wrapper for the Lab-Website Docker runner
REM ================================================================
REM  • Builds the image (if needed) from .\.docker\Dockerfile
REM  • Starts an interactive container that exposes
REM      http://localhost:4000  (site)
REM      http://localhost:35729 (LiveReload)
REM  • Any extra arguments you pass to run.bat are forwarded to the
REM    container entry-point (e.g. `run -- bundle exec rake clean`)
REM ================================================================

REM ----------- 1.  Config ----------------------------------------
set "IMAGE=lab-website-renderer:latest"
set "CONTAINER=lab-website-renderer"

REM ----------- 2.  Optional --platform flag ----------------------
REM Usage: run --platform linux/amd64
set "PLATFORM="
if /I "%~1"=="--platform" (
    if "%~2"=="" (
        echo Expected value after --platform
        exit /b 1
    )
    set "PLATFORM=--platform %~2"
    shift
    shift
)

REM ----------- 3.  Build the image -------------------------------
echo.
echo === Building Docker image %IMAGE% …
docker build %PLATFORM% --tag %IMAGE% --file .\.docker\Dockerfile .
if errorlevel 1 (
    echo Docker build failed – stopping.
    exit /b %errorlevel%
)

REM ----------- 4.  Determine host working dir -------------------
set "WORKDIR=%cd%"
REM If your Docker install balks at back-slashes, uncomment next line:
REM set "WORKDIR=%WORKDIR:\=/%" 

REM ----------- 5.  Run the container ----------------------------
echo.
echo === Running %CONTAINER% …
docker run %PLATFORM% ^
    --name %CONTAINER% ^
    --init ^
    --rm ^
    -it ^
    -p 4000:4000 ^
    -p 35729:35729 ^
    -v "%WORKDIR%:/usr/src/app" ^
    %IMAGE% %*
echo.
echo Container stopped.

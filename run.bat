@echo off
chcp 65001 > nul
setlocal

rem --------------------------------------------------------
rem .env から設定を読み込む
rem --------------------------------------------------------
set "VENV_PATH=.\.venv"

if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "line=%%A"
        if not "!line:~0,1!"=="#" (
            set "%%A=%%B"
        )
    )
)

set "IN_DIR=in"
set "OUT_DIR=out"

echo 【設定】
echo   仮想環境: %VENV_PATH%
echo   入力フォルダ: %IN_DIR%
echo   出力フォルダ: %OUT_DIR%
echo.

rem --------------------------------------------------------
rem 入力フォルダの確認
rem --------------------------------------------------------
if not exist "%IN_DIR%\" (
    echo [エラー] 入力フォルダ '%IN_DIR%' が存在しません。
    exit /b 1
)

set "CSV_COUNT=0"
for %%f in ("%IN_DIR%\*.csv") do set /a CSV_COUNT+=1
if %CSV_COUNT%==0 (
    echo [エラー] '%IN_DIR%' フォルダにCSVファイルが存在しません。
    exit /b 1
)
echo 入力CSVファイル数: %CSV_COUNT% 件

rem --------------------------------------------------------
rem 出力フォルダの作成
rem --------------------------------------------------------
if not exist "%OUT_DIR%\" mkdir "%OUT_DIR%"

rem --------------------------------------------------------
rem 仮想環境の作成（初回のみ）
rem --------------------------------------------------------
if not exist "%VENV_PATH%\" (
    echo 仮想環境を作成しています: %VENV_PATH%
    python -m venv "%VENV_PATH%"
    if errorlevel 1 (
        echo [エラー] 仮想環境の作成に失敗しました。Python がインストールされているか確認してください。
        exit /b 1
    )
)

rem --------------------------------------------------------
rem 仮想環境の有効化
rem --------------------------------------------------------
call "%VENV_PATH%\Scripts\activate.bat"
if errorlevel 1 (
    echo [エラー] 仮想環境の有効化に失敗しました。
    exit /b 1
)
echo 仮想環境を有効化しました:
python --version

rem --------------------------------------------------------
rem 依存パッケージのインストール
rem --------------------------------------------------------
echo 依存パッケージをインストールしています...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [エラー] パッケージのインストールに失敗しました。
    exit /b 1
)

rem --------------------------------------------------------
rem 実行
rem --------------------------------------------------------
echo.
echo 異常検知を開始します...
python main.py

echo.
echo 完了。結果は '%OUT_DIR%' フォルダを確認してください。

endlocal

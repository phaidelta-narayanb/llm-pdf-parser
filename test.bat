@echo off

REM python -m ragtest -o markuplm-test.xlsx -r TestLangchainSift .\tests\test_dummy.yml
python -m ragtest -o ollama-test.xlsx -m ollama/ChatOllama/gemma:2b .\tests\test_dummy.yml

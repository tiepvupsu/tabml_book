---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.8.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Làm sạch dữ liệu

Sau bước EDA, ta có cái nhìn đầu tiên về phân bố của các trường dữ liệu.
Việc cần làm tiếp theo là làm sạch dữ liệu bằng cách xử lý các giá trị ngoại lệ hoặc giá trị bị khuyết.
Ngoài ra, do đặc tính cửa việc thu thập dữ liệu, các giá trị như nhau có thể được lưu trong cơ sở dữ liệu dưới dạng khác nhau hoặc có lỗi chính tả trong các dữ liệu dạng hạng mục.
Dữ liệu dạng số và dạng hạng mục cần có những cách xử lý khác nhau.

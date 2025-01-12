'''
Author: SheathedSharp z404878860@163.com
Date: 2025-01-12 12:35:51
'''
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
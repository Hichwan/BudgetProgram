Simple Budget Program designed to take user input for certain things such as month, category, and the amount to add to the expense.

&nbsp;&nbsp;&nbsp;&nbsp;-It will ask to confirm each time after the month, category, and expense if that is correct. If it is given an invalid response, it will loop and ask again.

&nbsp;&nbsp;&nbsp;&nbsp;-The amount added to the expense will always be converted to a positive float value. If an expense for that month and category has already been added, it will combine the two expenses.

&nbsp;&nbsp;&nbsp;&nbsp;-The information/expenses are stored via SQLite database to a file within kaggle at /kaggle/working/Budget.db called Budget.db

&nbsp;&nbsp;&nbsp;&nbsp;-It has two ways to display the database: via a table or pie chart.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-The pie chart is specifically color coded for each expense and is labeled with the value being displayed as both a floating number and percent.

import os
import time
import win32com.client as win32
import pythoncom
import psutil

# 1. Close all Excel processes
def close_excel_processes():
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] and 'EXCEL.EXE' in proc.info['name'].upper():
            proc.kill()
    print("All Excel processes terminated.")

# 2. Define file path and sheet passwords
file_path = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\Snake Bite Data.xlsx"
sheet_passwords = {
    "Master Data": "masterdata2025!",
    "DropDowns": "dropdowns2025!",
    "CMCH": "cmch2020",
    "SOMCH": "somch2020",
    "SZMCH": "szmch2021",
    "RMCH": "rmch2021",
    "DMCH": "dmch2022",
    "KMCH": "kmch2022",
    "JMCH": "jmch2023",
    "RpMCH": "rpmch2023",
    "MMCH": "mmch2023",
    "SBMCH": "sbmch2024"
}

def unprotect_excel_sheets(file_path, passwords):
    pythoncom.CoInitialize()  # Initialize COM library for threading

    try:
        # Start Excel
        excel = win32.gencache.EnsureDispatch('Excel.Application')
        excel.Visible = True

        workbook = excel.Workbooks.Open(file_path)

        for sheet in workbook.Sheets:
            sheet_name = sheet.Name
            if sheet_name in passwords:
                try:
                    sheet.Unprotect(passwords[sheet_name])
                    print(f"Successfully unprotected sheet: {sheet_name}")
                except Exception as e:
                    print(f"Failed to unprotect {sheet_name} - wrong password?")
            else:
                print(f"No password found for sheet: {sheet_name}")

        workbook.Save()
        print("All sheets unprotected and file saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        try:
            workbook.Close(SaveChanges=0)
            excel.Quit()
            del workbook
            del excel
        except:
            pass
        pythoncom.CoUninitialize()
        print("Excel application closed and COM objects released.")

# === Execution ===
if __name__ == "__main__":
    close_excel_processes()
    unprotect_excel_sheets(file_path, sheet_passwords)
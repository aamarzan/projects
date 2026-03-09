import os
import time
import win32com.client as win32
import pythoncom
import psutil

def close_excel_processes():
    print("Checking for running Excel processes...")
    found_process = False
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'] and 'EXCEL.EXE' in proc.info['name'].upper():
                proc.kill()
                print(f"Terminated Excel process with PID: {proc.pid}")
                found_process = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    if not found_process:
        print("No running Excel processes found.")
    else:
        print("All Excel processes terminated.")
        time.sleep(2)

file_path = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\Unlocking Excels\NIHR RIGHT4 Methanol Dashboard.xlsm"
sheet_passwords = {
    "Dashboard": "dashboard2025!", "Master Data": "masterdata2025!", "CMCH": "cmch2020",
    "SOMCH": "somch2020", "SZMCH": "szmch2021", "RMCH": "rmch2021", "DMCH": "dmch2022",
    "PGIMER": "pgimer!", "SGRDUHS": "sgrduhs!", "GGSMC": "ggsmc!", "Daily": "daily2025!",
    "Monthly": "monthly2025!", "All Site": "allsite2025!", "Target": "target2025!", "True (+)": "true2025!"
}

def unprotect_excel_sheets(file_path, passwords):
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return

    pythoncom.CoInitialize()
    excel = None
    workbook = None

    try:
        excel = win32.Dispatch('Excel.Application')
        # Use True for debugging to see pop-ups, False for normal execution
        excel.Visible = True
        
        # --- NEW LINES TO PREVENT POP-UPS ---
        excel.DisplayAlerts = False
        excel.AskToUpdateLinks = False # Prevents the "update links" dialog
        excel.EnableEvents = False     # Prevents Auto_Open macros from running
        # ------------------------------------

        print(f"Opening workbook: {file_path}")
        # When opening, add parameters to be more explicit
        workbook = excel.Workbooks.Open(file_path, UpdateLinks=0, ReadOnly=False)


        # Give Excel a moment to fully load the workbook before proceeding
        time.sleep(3)

        for sheet in workbook.Sheets:
            # ... (rest of the loop is the same) ...
            sheet_name = sheet.Name
            if sheet_name in passwords:
                try:
                    sheet.Unprotect(passwords[sheet_name])
                    print(f"Successfully unprotected sheet: '{sheet_name}'")
                except Exception as e:
                    print(f"Could not unprotect sheet '{sheet_name}'. Wrong password or sheet not protected? Error: {e}")
            else:
                print(f"Skipping sheet '{sheet_name}': No password provided.")
        
        print("Saving the unprotected workbook...")
        workbook.Save()
        print("Workbook saved successfully.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        try:
            if workbook:
                # Re-enable events before closing if you disabled them
                excel.EnableEvents = True
                workbook.Close(SaveChanges=False)
            if excel:
                excel.Quit()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        del workbook
        del excel
        pythoncom.CoUninitialize()
        print("Excel application closed and COM objects released.")

if __name__ == "__main__":
    close_excel_processes()
    unprotect_excel_sheets(file_path, sheet_passwords)
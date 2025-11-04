import os


class TaskManager:
    def __init__(self, tasks):
        """
        tasks: list of task objects, each with process_frame(frame) and display_label() methods
        """
        self.tasks = tasks
        self.current_task_idx = 0  # Default to first task

    def process_frame(self, frame, key_char):
        """
        frame: current video frame
        key_char: character representing key pressed (e.g., ord('1'), ord('2'), ...)
        """
        # Only process key if it's valid
        if key_char is not None and key_char >= 0:
            key_str = chr(key_char)
            if key_str.isdigit():
                idx = int(key_str) - 1
                if 0 <= idx < len(self.tasks):
                    self.current_task_idx = idx
            elif key_str.lower() == "d":
                current_debug = os.getenv("DEBUG", "0")
                os.environ["DEBUG"] = "0" if current_debug == "1" else "1"
                print(f"Debug mode: {'ON' if os.environ['DEBUG'] == '1' else 'OFF'}")
        frame = self.tasks[self.current_task_idx].display_label(frame)
        return self.tasks[self.current_task_idx].process_frame(frame)

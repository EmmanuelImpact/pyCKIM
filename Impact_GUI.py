import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors
import webbrowser

# Ensure Matplotlib uses Tkinter-compatible backend
import matplotlib
matplotlib.use("TkAgg")

ctk.set_appearance_mode("light")  # Options: "dark", "light", "system"
ctk.set_default_color_theme("blue")

def run_simulation(E, nu, sigma_y, H_iso, H_kin, mix_ratio):
    n_steps = 10000
    epsilon_total = np.array([0.01, 0.005, 0.003, 0, 0, 0])
    delta_epsilon = epsilon_total / n_steps

    G = E / (2 * (1 + nu))  # Shear modulus
    K = E / (3 * (1 - 2 * nu))  # Bulk modulus

    C = E / ((1 + nu) * (1 - 2 * nu)) * np.array([  
        [1 - nu, nu, nu, 0, 0, 0],
        [nu, 1 - nu, nu, 0, 0, 0],
        [nu, nu, 1 - nu, 0, 0, 0],
        [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
        [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
        [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]
    ]) 

    epsilon = np.zeros(6)
    epsilon_p = np.zeros(6)
    sigma = np.zeros(6)
    alpha = np.zeros(6)
    sigma_iso = 0

    strain = np.zeros((6, n_steps))
    stress = np.zeros((6, n_steps))
    von_mises = np.zeros(n_steps)

    for i in range(n_steps):
        epsilon += delta_epsilon
        sigma_trial = np.dot(C, epsilon - epsilon_p)
        dev_sigma_trial = sigma_trial - np.mean(sigma_trial[:3]) * np.array([1, 1, 1, 0, 0, 0])
        sigma_eff_trial = dev_sigma_trial - alpha
        J2 = 0.5 * np.dot(sigma_eff_trial, sigma_eff_trial)
        sigma_eq = np.sqrt(3 * J2)
        f_yield = sigma_eq - (sigma_y + sigma_iso)

        if f_yield <= 0:
            sigma = sigma_trial
        else:
            delta_gamma = f_yield / (3 * G + H_iso * (1 - mix_ratio) + H_kin * mix_ratio)
            normal = sigma_eff_trial / sigma_eq
            epsilon_p += delta_gamma * normal
            alpha += 2 * G * mix_ratio * delta_gamma * normal
            sigma_iso += H_iso * (1 - mix_ratio) * delta_gamma
            sigma = sigma_trial - 2 * G * delta_gamma * normal

        strain[:, i] = epsilon
        stress[:, i] = sigma
        von_mises[i] = sigma_eq

    return strain, stress, von_mises, alpha, sigma_iso

def create_input_field(parent, label_text):
    """Create a standardized input field with label"""
    frame = ctk.CTkFrame(parent)
    frame.pack(fill="x", padx=10, pady=5)
    
    label = ctk.CTkLabel(frame, text=label_text, font=("Segoe UI", 14, "bold"), width=250, anchor="w")
    label.pack(side="left", padx=10)
    
    entry = ctk.CTkEntry(frame, font=("Segoe UI", 12), width=250)
    entry.pack(side="right", padx=10, fill="x", expand=True)
    
    return entry

def on_run_simulation():
    try:
        # Get input values
        E = float(entry_E.get().strip())
        nu = float(entry_nu.get().strip())
        sigma_y = float(entry_sigma_y.get().strip())
        H_iso = float(entry_H_iso.get().strip())
        H_kin = float(entry_H_kin.get().strip())
        mix_ratio = float(entry_mix_ratio.get().strip())

        # Validate mix_ratio to ensure it is between 0 and 1
        if not (0 <= mix_ratio <= 1):
            messagebox.showerror("Input Error", "Mixing Ratio must be between 0 and 1!")
            return

        # Run the simulation
        strain, stress, von_mises, alpha, sigma_iso = run_simulation(E, nu, sigma_y, H_iso, H_kin, mix_ratio)
        n_steps = strain.shape[1]

        # Clear previous results
        for widget in results_frame.winfo_children():
            widget.destroy()

        # Create figure with appropriate size
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
        plt.subplots_adjust(hspace=0.4, wspace=0.4)  # More spacing to avoid label overlap

        # Strain evolution
        axs[0, 0].plot(range(n_steps), strain[0, :], 'b', label=r'$\epsilon_{xx}$')
        axs[0, 0].plot(range(n_steps), strain[1, :], 'r', label=r'$\epsilon_{yy}$')
        axs[0, 0].plot(range(n_steps), strain[2, :], 'g', label=r'$\epsilon_{zz}$')
        axs[0, 0].set_xlabel('Step', fontsize=9)
        axs[0, 0].set_ylabel('Strain', fontsize=9)
        axs[0, 0].legend(fontsize=8)
        axs[0, 0].set_title('Strain Evolution', fontsize=10)
        axs[0, 0].grid(True)
        mplcursors.cursor(axs[0, 0], hover=True)

        # Stress evolution
        axs[0, 1].plot(range(n_steps), stress[0, :], 'b', label=r'$\sigma_{xx}$')
        axs[0, 1].plot(range(n_steps), stress[1, :], 'r', label=r'$\sigma_{yy}$')
        axs[0, 1].plot(range(n_steps), stress[2, :], 'g', label=r'$\sigma_{zz}$')
        axs[0, 1].set_xlabel('Step', fontsize=9)
        axs[0, 1].set_ylabel('Stress (MPa)', fontsize=9)
        axs[0, 1].legend(fontsize=8)
        axs[0, 1].set_title('Stress Evolution', fontsize=10)
        axs[0, 1].grid(True)
        mplcursors.cursor(axs[0, 1], hover=True)

        # Von Mises stress evolution
        axs[1, 0].plot(range(n_steps), von_mises, 'm', label='Von Mises Stress')
        axs[1, 0].set_xlabel('Step', fontsize=9)
        axs[1, 0].set_ylabel('Von Mises Stress (MPa)', fontsize=9)
        axs[1, 0].legend(fontsize=8)
        axs[1, 0].set_title('Von Mises Stress Evolution', fontsize=10)
        axs[1, 0].grid(True)
        mplcursors.cursor(axs[1, 0], hover=True)

        # Yield Surface Evolution
        theta = np.linspace(0, 2 * np.pi, 100)
        initial_radius = sigma_y
        x_initial = initial_radius * np.cos(theta)
        y_initial = initial_radius * np.sin(theta)
        axs[1, 1].plot(x_initial, y_initial, 'k--', label='Initial Yield Surface')

        if mix_ratio == 0:
            final_iso_radius = sigma_y + sigma_iso
            axs[1, 1].plot(final_iso_radius * np.cos(theta), final_iso_radius * np.sin(theta), 'g-', label='Isotropic Yield Surface')
        elif mix_ratio == 1:
            axs[1, 1].plot(alpha[0] + initial_radius * np.cos(theta), alpha[1] + initial_radius * np.sin(theta), 'r-', label='Kinematic Yield Surface')
        else:
            final_iso_radius = sigma_y + sigma_iso
            mixed_x = alpha[0] + final_iso_radius * np.cos(theta)
            mixed_y = alpha[1] + final_iso_radius * np.sin(theta)
            axs[1, 1].plot(final_iso_radius * np.cos(theta), final_iso_radius * np.sin(theta), 'g-', label='Isotropic Yield Surface')
            axs[1, 1].plot(alpha[0] + initial_radius * np.cos(theta), alpha[1] + initial_radius * np.sin(theta), 'r-', label='Kinematic Yield Surface')
            axs[1, 1].plot(mixed_x, mixed_y, 'b-', linewidth=2, label='Mixed Hardening Yield Surface')

        axs[1, 1].set_title('Yield Surface Evolution', fontsize=10)
        axs[1, 1].set_xlabel(r'$\sigma_{xx} - \sigma_{yy}$', fontsize=9)
        axs[1, 1].set_ylabel(r'$\sigma_{zz}$', fontsize=9)
        axs[1, 1].legend(loc='upper right', fontsize=8, frameon=True)
        axs[1, 1].grid(True)
        axs[1, 1].axis('equal')
        mplcursors.cursor(axs[1, 1], hover=True)

        # Adjust layout to fit text properly
        fig.tight_layout()

        # Embed matplotlib figure in Tkinter
        canvas = FigureCanvasTkAgg(fig, results_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()

        # Create a frame above the plot area for the zoom slider
        zoom_frame = ctk.CTkFrame(results_frame, fg_color="blue")
        zoom_frame.place(relx=1.0, rely=0.05, anchor="ne", y=-25, x=-10)  # Adjusted downward for visibility

        # Zoom slider (longer and properly positioned)
        zoom_slider = ctk.CTkSlider(
            master=zoom_frame, 
            from_=50, 
            to=150, 
            number_of_steps=20,
            width=100  # Increased width for a longer slider
        )
        zoom_slider.set(100)
        zoom_slider.pack(side='right')

        # Function to handle zooming
        def update_zoom(value):
            scale_factor = float(value) / 100
            fig.set_size_inches(10 * scale_factor, 6 * scale_factor)
            canvas.draw()

        zoom_slider.configure(command=update_zoom)

        # Show simulation completed message
        messagebox.showinfo("Simulation Completed", "The simulation has been successfully completed! Click the Results tab to view the outcome of this simulation!")

    except ValueError as e:
        messagebox.showerror("Input Error", f"Invalid input: {e}")
        
def clear_inputs():
    # Clear all input fields
    entry_E.delete(0, tk.END)
    entry_nu.delete(0, tk.END)
    entry_sigma_y.delete(0, tk.END)
    entry_H_iso.delete(0, tk.END)
    entry_H_kin.delete(0, tk.END)
    entry_mix_ratio.delete(0, tk.END)
    
    # Also clear the results if inputs are cleared
    for widget in results_frame.winfo_children():
        widget.destroy()

def toggle_theme():
    # Get current theme and switch to opposite
    current_theme = ctk.get_appearance_mode().lower()
    new_theme = "dark" if current_theme == "light" else "light"
    
    # Update the appearance mode
    ctk.set_appearance_mode(new_theme)
    
    # Update the label text
    btn_theme.configure(text=f"Current Theme: {new_theme.capitalize()}")


# Create main window
root = ctk.CTk()
root.title("Material Behavior Simulator")
root.geometry("1300x600")  # Increased initial window size
root.minsize(1300, 600)  # Set a minimum window size

# Create a main container frame to limit content width
main_container = ctk.CTkFrame(root, fg_color="transparent")
main_container.pack(expand=True, fill="both", padx=200)  # Add 100px padding on each side

# Create tabs within the main container
tab_control = ctk.CTkTabview(main_container)
tab_control.pack(expand=True, fill="both", padx=20, pady=10)

input_tab = tab_control.add("Inputs")
results_tab = tab_control.add("Results")
help_tab = tab_control.add("Help")

# Results Tab
results_frame = ctk.CTkFrame(results_tab)
results_frame.pack(fill="both", expand=True, padx=20, pady=10)

# Input Section - Now with controlled width
input_container = ctk.CTkFrame(input_tab, fg_color="transparent")
input_container.pack(expand=True, fill="both", padx=40, pady=20)  # Increased padding

# Create a frame to center and limit the width of input fields
input_width_limiter = ctk.CTkFrame(input_container, fg_color="transparent")
input_width_limiter.pack(expand=True, fill="both", padx=20)

# Title
title_label = ctk.CTkLabel(input_width_limiter, text="Material Parameters", font=("Segoe UI", 18, "bold"))
title_label.pack(pady=(10, 20))

# Create input fields inside the width limiter
entry_E = create_input_field(input_width_limiter, "Young's Modulus (E in MPa):")
entry_E.configure(border_width=1, height=30)

entry_nu = create_input_field(input_width_limiter, "Poisson's Ratio (ν):")
entry_nu.configure(border_width=1, height=30)

entry_sigma_y = create_input_field(input_width_limiter, "Initial Yield Stress (σ_y in MPa):")
entry_sigma_y.configure(border_width=1, height=30)

entry_H_iso = create_input_field(input_width_limiter, "Isotropic Hardening Modulus (H_iso):")
entry_H_iso.configure(border_width=1, height=30)

entry_H_kin = create_input_field(input_width_limiter, "Kinematic Hardening Modulus (H_kin):")
entry_H_kin.configure(border_width=1, height=30)

entry_mix_ratio = create_input_field(input_width_limiter, "Mixing Ratio:")
entry_mix_ratio.configure(border_width=1, height=30)

# Button Container with controlled width
button_container = ctk.CTkFrame(input_width_limiter, fg_color="transparent")
button_container.pack(fill="x", pady=20)

# Add a frame to center the buttons and control their width
button_width_limiter = ctk.CTkFrame(button_container, fg_color="transparent")
button_width_limiter.pack(expand=True, padx=40)

btn_run = ctk.CTkButton(
    button_width_limiter,
    text="Run Simulation",
    command=on_run_simulation,
    fg_color="#2563EB",
    hover_color="#1E3A8A",
    font=("Segoe UI", 14, "bold"),
    width=200  # Fixed width for buttons
)
btn_run.pack(side="left", pady=(50, 0), padx=10)
btn_run.configure(height=40)

btn_clear = ctk.CTkButton(
    button_width_limiter,
    text="Clear Inputs",
    command=clear_inputs,
    fg_color="#E74C3C",
    hover_color="#C0392B",
    font=("Segoe UI", 14, "bold"),
    width=200  # Fixed width for buttons
)
btn_clear.pack(side="left", pady=(50, 0), padx=10)
btn_clear.configure(height=40)

# Theme toggle button with controlled width
theme_button_container = ctk.CTkFrame(main_container, fg_color="transparent")
theme_button_container.pack(fill="x", padx=20, pady=10)

btn_theme = ctk.CTkButton(
    theme_button_container,
    text="Toggle Theme",
    command=toggle_theme,
    fg_color="#2563EB",
    hover_color="#1E3A8A",
    width=200  # Fixed width for consistency
)
btn_theme.pack(side="left", padx=10)

# Help Section
def open_documentation():
    webbrowser.open("https://raw.githubusercontent.com/YourUsername/YourRepo/main/docs/user_documentation.pdf")

def open_github():
    webbrowser.open("https://github.com/YourUsername/YourRepo")  # Replace with the actual GitHub link

# Function to toggle theme and update text color
def toggle_theme():
    # Get current theme and switch to the opposite
    current_theme = ctk.get_appearance_mode()
    new_theme = "Dark" if current_theme == "Light" else "Light"

    # Set the new theme
    ctk.set_appearance_mode(new_theme)

    # Determine the correct text color based on the theme
    text_color = "white" if new_theme == "Dark" else "black"

    # Apply the text color change to help_label
    if help_label.winfo_exists():  # Ensure the label exists before configuring
        help_label.configure(text_color=text_color)

    # Update the theme button text
    btn_theme.configure(text=f"Current Theme: {new_theme}")

# Help Section UI
help_frame = ctk.CTkFrame(help_tab)
help_frame.place(relx=0.5, rely=0.5, anchor="center")

help_label = ctk.CTkLabel(
    help_frame,
    text="📖 Help & User Guide",
    font=("Segoe UI", 18, "bold"),
    text_color="black"  # Default color for Light mode
)
help_label.pack(pady=10)

download_button = ctk.CTkButton(
    help_frame,
    text="📥 Download User Documentation",
    command=open_documentation,
    font=("Segoe UI", 14, "bold"),
    fg_color="#007BFF",
    hover_color="#0056b3"
)
download_button.pack(pady=10)

# GitHub Source Code Link
github_link = ctk.CTkButton(
    help_frame,
    text="🔗 Access the Source Code",
    command=open_github,
    font=("Segoe UI", 12, "underline"),
    fg_color="transparent",  # Make it look like a link
    text_color="#1E90FF",  # Blue link color
    hover_color="#0056b3"  # Darker blue on hover
)
github_link.pack(pady=(10, 20))  # Extra padding at the bottom

# Ensure btn_theme is already defined in your GUI (DO NOT CREATE A NEW ONE)
btn_theme.configure(command=toggle_theme)  # Assign toggle_theme function to your existing button

# Ensure it expands properly in the window
#help_content.pack(expand=True, padx=20, pady=20)

# Run main loop
root.mainloop()
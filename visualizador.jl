using GLMakie
using NPZ
using Printf

include("utils/gru.jl")
#include("Slicer3D.jl")   # or paste your Slicer3D function directly here

function visualize_Bfield(; filepath="data/simulaciones/B_field.npz", component="By")
    # Load data
    data = npzread(filepath)

    Bx = data["Bx"]
    By = data["By"]
    Bz = data["Bz"]

    # Select component
    B = component == "Bx" ? Bx :
        component == "By" ? By :
        component == "Bz" ? Bz :
        error("Component must be Bx, By or Bz")

    println("Loaded component: $component")
    println("Shape: ", size(B))

    #include("utils/gru.jl")
    #@cuda threads=threads blocks=blocks shmem=shmem kernel_fused_B!(R, P, M, B, n, m)
    #B_res = Array(B')

    #XX = [xi for xi in gx, yi in gy, zi in gz]
    #mask = trues(size(XX))
    #By = zeros(size(XX))
    #@allowscalar begin
    #    By[mask] = B_res[2,:] .* -1000 # mT, el menos es unicamente para invertir los colores del heatmap
    #    fig = Figure(size=(600,600))
    #    saxi = Slicer3D(fig,By,zoom=3)
    #    display(fig)
    #end

    # Create figure
    fig = Figure(size=(700, 600))

    # Launch slicer
    Slicer3D(fig, B, zoom=2)

    display(fig)
    
end

# Run it
visualize_Bfield(component="By")
readline()
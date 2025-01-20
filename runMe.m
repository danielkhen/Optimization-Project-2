question = input("Enter question number to run:");

try
    run("Q"+question+".m");
catch e
    if e.identifier == "MATLAB:run:FileNotFound"
        error("Question doesn't have code");
    end
end
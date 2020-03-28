library('socialmixr')
data($survey_source)
age_dist <- read.csv("$age_dist_file_path")
age_limits = c()
for(v in age_dist['lower.age.limit']) { age_limits <- v }

m <- contact_matrix($survey_source,
                    countries = "$country",
                    age.limits = age_limits,
                    n=$num_sample,
                    survey.pop=age_dist,
                    weigh.dayofweek=$weight_by_dayofweek)
mr <- Reduce("+", lapply(m$matrices, function(x) {x$matrix})) / length(m$matrices)

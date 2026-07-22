from Main import rank_resumes_tfidf

def test_more_relevant_resume_scores_higher():
    job_description = "Looking for a Python developer with experience in Flask, REST APIs, and PostgreSQL."
    strong_match = "Experienced Python developer skilled in Flask, REST API design, and PostgreSQL database management."
    weak_match = "Chef with five years of experience in Italian cuisine and restaurant management."

    scores = rank_resumes_tfidf(job_description, [strong_match, weak_match])

    assert len(scores) == 2
    assert 0 <= scores[0] <= 100
    assert 0 <= scores[1] <= 100
    assert scores[0] > scores[1]
